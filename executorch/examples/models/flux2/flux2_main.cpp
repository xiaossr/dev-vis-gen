/*
 * FLUX.2-klein-4B on-device inference using ExecuTorch XNNPACK.
 *
 * Pipeline:  text_encoder.pte → transformer.pte → vae_decoder.pte
 *
 * Token IDs are pre-tokenized on the host via prepare_mobile.py.
 * All neural-network execution happens on-device through .pte files.
 *
 * Output is a PPM image (convert to PNG with: convert output.ppm output.png).
 */

 #include <executorch/extension/module/module.h>
 #include <executorch/extension/tensor/tensor.h>
 #include <gflags/gflags.h>
 
 #include <algorithm>
 #include <chrono>
 #include <cmath>
 #include <cstdint>
 #include <cstdio>
 #include <cstring>
 #include <random>
 #include <string>
 #include <vector>
 
 using executorch::extension::Module;
 using executorch::extension::from_blob;
 using executorch::aten::ScalarType;
 using executorch::runtime::EValue;
 using executorch::runtime::Error;
 
 // ---------------------------------------------------------------------------
 // gflags
 // ---------------------------------------------------------------------------
 
 DEFINE_string(model_dir, "", "Directory with .pte files and bn_*.bin");
 DEFINE_string(tokens, "", "Tokenized prompt binary (from prepare_mobile.py)");
 DEFINE_string(output, "output.ppm", "Output image path (PPM format)");
 DEFINE_int32(height, 512, "Image height");
 DEFINE_int32(width, 512, "Image width");
 DEFINE_int32(steps, 4, "Number of denoising steps");
 DEFINE_int32(seed, 42, "Random seed");
 DEFINE_int32(vae_sf, 8, "VAE scale factor");
 DEFINE_int32(in_channels, 128, "Transformer in_channels (post-patchify)");
 DEFINE_int32(max_seq_len, 512, "Max text sequence length");
 
 // ---------------------------------------------------------------------------
 // Timing helper
 // ---------------------------------------------------------------------------
 
 static double now_sec() {
   using namespace std::chrono;
   return duration<double>(steady_clock::now().time_since_epoch()).count();
 }
 
 // ---------------------------------------------------------------------------
 // Binary I/O
 // ---------------------------------------------------------------------------
 
 struct TokenData {
   std::vector<int64_t> input_ids;
   std::vector<int64_t> attention_mask;
   int seq_len;
 };
 
 static TokenData load_tokens(const std::string& path) {
   FILE* f = fopen(path.c_str(), "rb");
   if (!f) {
     fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
     exit(1);
   }
   int32_t seq_len;
   fread(&seq_len, sizeof(int32_t), 1, f);
 
   TokenData td;
   td.seq_len = seq_len;
   td.input_ids.resize(seq_len);
   td.attention_mask.resize(seq_len);
   fread(td.input_ids.data(), sizeof(int64_t), seq_len, f);
   fread(td.attention_mask.data(), sizeof(int64_t), seq_len, f);
   fclose(f);
   return td;
 }
 
 static std::vector<float> load_floats_bin(const std::string& path) {
   FILE* f = fopen(path.c_str(), "rb");
   if (!f) return {};
   int32_t n;
   fread(&n, sizeof(int32_t), 1, f);
   std::vector<float> v(n);
   fread(v.data(), sizeof(float), n, f);
   fclose(f);
   return v;
 }
 
 // ---------------------------------------------------------------------------
 // Sigma schedule  (FlowMatchEulerDiscreteScheduler with dynamic time-shift)
 // ---------------------------------------------------------------------------
 
 static float compute_mu(int image_seq_len, int num_steps) {
   constexpr float a1 = 8.73809524e-05f, b1 = 1.89833333f;
   constexpr float a2 = 0.00016927f, b2 = 0.45666666f;
   if (image_seq_len > 4300)
     return a2 * image_seq_len + b2;
   float m_200 = a2 * image_seq_len + b2;
   float m_10 = a1 * image_seq_len + b1;
   float a = (m_200 - m_10) / 190.0f;
   float b = m_200 - 200.0f * a;
   return a * num_steps + b;
 }
 
 static std::vector<float> build_sigmas(int num_steps, int image_seq_len) {
   std::vector<float> sigmas(num_steps);
   if (num_steps == 1) {
     sigmas[0] = 1.0f;
   } else {
     float end = 1.0f / num_steps;
     for (int i = 0; i < num_steps; i++)
       sigmas[i] = 1.0f + (float)i * (end - 1.0f) / (num_steps - 1);
   }
   float mu = compute_mu(image_seq_len, num_steps);
   float emu = expf(mu);
   for (auto& s : sigmas)
     s = emu / (emu + (1.0f / s - 1.0f));
   sigmas.push_back(0.0f);
   return sigmas;
 }
 
 // ---------------------------------------------------------------------------
 // Positional-ID generation  (4-D coordinates: T, H, W, L)
 // ---------------------------------------------------------------------------
 
 static std::vector<float> make_img_ids(int patch_h, int patch_w) {
   int N = patch_h * patch_w;
   std::vector<float> ids(N * 4, 0.0f);
   int idx = 0;
   for (int h = 0; h < patch_h; h++) {
     for (int w = 0; w < patch_w; w++) {
       ids[idx * 4 + 1] = static_cast<float>(h);
       ids[idx * 4 + 2] = static_cast<float>(w);
       idx++;
     }
   }
   return ids;
 }
 
 static std::vector<float> make_txt_ids(int seq_len) {
   std::vector<float> ids(seq_len * 4, 0.0f);
   for (int s = 0; s < seq_len; s++)
     ids[s * 4 + 3] = static_cast<float>(s);
   return ids;
 }
 
 // ---------------------------------------------------------------------------
 // Latent packing / unpacking / unpatchify / BN
 // ---------------------------------------------------------------------------
 
 // (C, H, W) → (H*W, C)
 static std::vector<float> pack_chw_to_nc(
     const std::vector<float>& src, int C, int H, int W) {
   std::vector<float> dst(H * W * C);
   for (int c = 0; c < C; c++)
     for (int h = 0; h < H; h++)
       for (int w = 0; w < W; w++)
         dst[(h * W + w) * C + c] = src[c * H * W + h * W + w];
   return dst;
 }
 
 // (N, C) + (N, 4) → (C, H, W)   using H/W from the IDs
 static std::vector<float> unpack_nc_to_chw(
     const float* data, const float* ids, int N, int C, int H, int W) {
   std::vector<float> out(C * H * W, 0.0f);
   for (int i = 0; i < N; i++) {
     int h = static_cast<int>(ids[i * 4 + 1]);
     int w = static_cast<int>(ids[i * 4 + 2]);
     for (int c = 0; c < C; c++)
       out[c * H * W + h * W + w] = data[i * C + c];
   }
   return out;
 }
 
 // x = x * sqrt(var + eps) + mean   (per-channel, CHW layout)
 static void bn_unnormalize(
     float* data, const float* mean, const float* var,
     int C, int H, int W, float eps) {
   for (int c = 0; c < C; c++) {
     float s = sqrtf(var[c] + eps);
     float m = mean[c];
     float* ch = data + c * H * W;
     for (int i = 0; i < H * W; i++)
       ch[i] = ch[i] * s + m;
   }
 }
 
 // (C*4, H2, W2) → (C, H2*2, W2*2)
 static std::vector<float> unpatchify(
     const float* src, int C4, int H2, int W2) {
   int C = C4 / 4;
   int H = H2 * 2;
   int W = W2 * 2;
   std::vector<float> dst(C * H * W);
   for (int c = 0; c < C; c++)
     for (int h = 0; h < H; h++)
       for (int w = 0; w < W; w++) {
         int src_c = c * 4 + (h % 2) * 2 + (w % 2);
         dst[c * H * W + h * W + w] = src[src_c * H2 * W2 + (h / 2) * W2 + (w / 2)];
       }
   return dst;
 }
 
 // ---------------------------------------------------------------------------
 // PPM image writer  (convert to PNG with:  convert output.ppm output.png)
 // ---------------------------------------------------------------------------
 
 static void save_ppm(
     const char* path, const float* chw, int H, int W) {
   FILE* f = fopen(path, "wb");
   if (!f) {
     fprintf(stderr, "ERROR: cannot write %s\n", path);
     return;
   }
   fprintf(f, "P6\n%d %d\n255\n", W, H);
   for (int h = 0; h < H; h++)
     for (int w = 0; w < W; w++)
       for (int c = 0; c < 3; c++) {
         float v = chw[c * H * W + h * W + w];
         v = std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);
         fputc(static_cast<unsigned char>(v * 255.0f), f);
       }
   fclose(f);
 }
 
 // ---------------------------------------------------------------------------
 // Main
 // ---------------------------------------------------------------------------
 
 int main(int argc, char** argv) {
   gflags::ParseCommandLineFlags(&argc, &argv, true);
 
   if (FLAGS_model_dir.empty() || FLAGS_tokens.empty()) {
     fprintf(stderr,
             "Usage: flux2_main --model_dir <dir> --tokens <prompt.bin> "
             "[--output output.ppm] [--steps 4] [--seed 42]\n");
     return 1;
   }
 
   const int height = FLAGS_height;
   const int width = FLAGS_width;
   const int vae_sf = FLAGS_vae_sf;
   const int in_ch = FLAGS_in_channels;
   const int seq_len = FLAGS_max_seq_len;
   const int num_steps = FLAGS_steps;
   const int seed = FLAGS_seed;
   constexpr float bn_eps = 1e-5f;
 
   int latent_h = 2 * (height / (vae_sf * 2));
   int latent_w = 2 * (width / (vae_sf * 2));
   int patch_h = latent_h / 2;
   int patch_w = latent_w / 2;
   int num_tokens = patch_h * patch_w;
   int lat_ch = in_ch / 4;
 
   printf("=== FLUX.2-klein on-device inference ===\n");
   printf("Resolution   : %dx%d\n", height, width);
   printf("Patch dims   : %dx%d  (%d tokens)\n", patch_h, patch_w, num_tokens);
   printf("Steps / seed : %d / %d\n", num_steps, seed);
   fflush(stdout);
 
   // ---- load tokens -------------------------------------------------------
   auto tok = load_tokens(FLAGS_tokens);
   printf("Tokens loaded : seq_len=%d\n", tok.seq_len);
   fflush(stdout);
 
   // ---- load BN stats -----------------------------------------------------
   auto bn_mean = load_floats_bin(FLAGS_model_dir + "/bn_mean.bin");
   auto bn_var = load_floats_bin(FLAGS_model_dir + "/bn_var.bin");
   bool has_bn = !bn_mean.empty() && !bn_var.empty();
   if (has_bn)
     printf("BN stats      : %zu channels\n", bn_mean.size());
   fflush(stdout);
 
   // ======================= 1. TEXT ENCODING ================================
   // Load text encoder, run it, then release to free memory before loading
   // the transformer.  This keeps peak memory to ~1 model at a time.
   printf("\n[1/4] Loading text_encoder.pte ...\n");
   fflush(stdout);
   double t0 = now_sec();
 
   std::vector<float> prompt_embeds;
   int joint_dim = 0;
   {
     Module text_encoder(FLAGS_model_dir + "/text_encoder.pte");
     printf("  loaded (%.1f s)\n", now_sec() - t0);
     fflush(stdout);
 
     auto ids_tp = from_blob(tok.input_ids.data(), {1, tok.seq_len}, ScalarType::Long);
     auto mask_tp = from_blob(tok.attention_mask.data(), {1, tok.seq_len}, ScalarType::Long);
 
     std::vector<EValue> te_inputs;
     te_inputs.push_back(*ids_tp);
     te_inputs.push_back(*mask_tp);
 
     printf("  running text encoder ...\n");
     fflush(stdout);
     t0 = now_sec();
     auto te_res = text_encoder.execute("forward", te_inputs);
     if (!te_res.ok()) {
       fprintf(stderr, "ERROR: text_encoder forward failed\n");
       return 1;
     }
     const auto& te_tensor = (*te_res)[0].toTensor();
     prompt_embeds.assign(
         te_tensor.const_data_ptr<float>(),
         te_tensor.const_data_ptr<float>() + te_tensor.numel());
     joint_dim = static_cast<int>(te_tensor.numel()) / seq_len;
     printf("  prompt_embeds: %d x %d  (%.1f s)\n", seq_len, joint_dim, now_sec() - t0);
     fflush(stdout);
   }
   // text_encoder is destroyed here — its memory is freed
   printf("  text encoder released\n");
   fflush(stdout);
 
   // ======================= 2. NOISE GENERATION =============================
   printf("\n[2/4] Noise generation\n");
   fflush(stdout);
   std::vector<float> noise(in_ch * patch_h * patch_w);
   {
     std::mt19937 rng(seed);
     std::normal_distribution<float> dist(0.0f, 1.0f);
     for (auto& x : noise)
       x = dist(rng);
   }
   auto latents = pack_chw_to_nc(noise, in_ch, patch_h, patch_w);
   noise.clear();
   noise.shrink_to_fit();
   printf("  latents: %d x %d\n", num_tokens, in_ch);
 
   auto sigmas = build_sigmas(num_steps, num_tokens);
   printf("  sigmas:");
   for (auto s : sigmas)
     printf(" %.4f", s);
   printf("\n");
 
   auto img_ids = make_img_ids(patch_h, patch_w);
   auto txt_ids = make_txt_ids(seq_len);
   fflush(stdout);
 
   // ======================= 3. DENOISING LOOP ===============================
   printf("\n[3/4] Loading transformer.pte ...\n");
   fflush(stdout);
   t0 = now_sec();
   {
     Module transformer(FLAGS_model_dir + "/transformer.pte");
     printf("  loaded (%.1f s)\n", now_sec() - t0);
     fflush(stdout);
 
     printf("  denoising (%d steps) ...\n", num_steps);
     fflush(stdout);
     double denoise_t0 = now_sec();
 
     for (int step = 0; step < num_steps; step++) {
       float sigma = sigmas[step];
       float sigma_next = sigmas[step + 1];
       std::vector<float> ts_buf = {sigma};
 
       auto hs_tp = from_blob(latents.data(), {1, num_tokens, in_ch}, ScalarType::Float);
       auto ehs_tp = from_blob(prompt_embeds.data(), {1, seq_len, joint_dim}, ScalarType::Float);
       auto ts_tp = from_blob(ts_buf.data(), {1}, ScalarType::Float);
       auto iid_tp = from_blob(img_ids.data(), {1, num_tokens, 4}, ScalarType::Float);
       auto tid_tp = from_blob(txt_ids.data(), {1, seq_len, 4}, ScalarType::Float);
 
       std::vector<EValue> tf_inputs;
       tf_inputs.push_back(*hs_tp);
       tf_inputs.push_back(*ehs_tp);
       tf_inputs.push_back(*ts_tp);
       tf_inputs.push_back(*iid_tp);
       tf_inputs.push_back(*tid_tp);
 
       t0 = now_sec();
       auto tf_res = transformer.execute("forward", tf_inputs);
       double step_sec = now_sec() - t0;
 
       if (!tf_res.ok()) {
         fprintf(stderr, "ERROR: transformer failed at step %d\n", step);
         return 1;
       }
       const auto& pred_t = (*tf_res)[0].toTensor();
       const float* pred = pred_t.const_data_ptr<float>();
 
       float dt = sigma_next - sigma;
       for (size_t j = 0; j < latents.size(); j++)
         latents[j] += dt * pred[j];
 
       printf("  step %d/%d  sigma %.4f -> %.4f  (%.2f s)\n",
              step + 1, num_steps, sigma, sigma_next, step_sec);
       fflush(stdout);
     }
     printf("  total denoise: %.1f s\n", now_sec() - denoise_t0);
     fflush(stdout);
   }
   // transformer is destroyed here — its memory is freed
   printf("  transformer released\n");
   fflush(stdout);
 
   // Free intermediate data no longer needed
   prompt_embeds.clear();
   prompt_embeds.shrink_to_fit();
   img_ids.clear();
   img_ids.shrink_to_fit();
   txt_ids.clear();
   txt_ids.shrink_to_fit();
 
   // ======================= 4. POST-PROCESSING ==============================
   printf("\n[4/4] Post-processing + VAE decode\n");
   fflush(stdout);
 
   auto spatial = unpack_nc_to_chw(
       latents.data(), make_img_ids(patch_h, patch_w).data(),
       num_tokens, in_ch, patch_h, patch_w);
   latents.clear();
   latents.shrink_to_fit();
   printf("  unpacked: %d x %d x %d\n", in_ch, patch_h, patch_w);
 
   if (has_bn) {
     bn_unnormalize(spatial.data(), bn_mean.data(), bn_var.data(),
                    in_ch, patch_h, patch_w, bn_eps);
     printf("  BN un-normalised\n");
   }
 
   auto lat_full = unpatchify(spatial.data(), in_ch, patch_h, patch_w);
   spatial.clear();
   spatial.shrink_to_fit();
   printf("  unpatchified: %d x %d x %d\n", lat_ch, latent_h, latent_w);
   fflush(stdout);
 
   // ---- VAE decode (load → run → release) ---------------------------------
   printf("  loading vae_decoder.pte ...\n");
   fflush(stdout);
   t0 = now_sec();
   {
     Module vae_decoder(FLAGS_model_dir + "/vae_decoder.pte");
     printf("  loaded (%.1f s)\n", now_sec() - t0);
     fflush(stdout);
 
     auto vae_tp = from_blob(lat_full.data(), {1, lat_ch, latent_h, latent_w}, ScalarType::Float);
 
     std::vector<EValue> vae_inputs;
     vae_inputs.push_back(*vae_tp);
 
     printf("  running VAE decoder ...\n");
     fflush(stdout);
     t0 = now_sec();
     auto vae_res = vae_decoder.execute("forward", vae_inputs);
     if (!vae_res.ok()) {
       fprintf(stderr, "ERROR: vae_decoder forward failed\n");
       return 1;
     }
     const auto& pix_t = (*vae_res)[0].toTensor();
     printf("  decoded: numel=%zu  (%.1f s)\n", (size_t)pix_t.numel(), now_sec() - t0);
     fflush(stdout);
 
     save_ppm(FLAGS_output.c_str(), pix_t.const_data_ptr<float>(), height, width);
   }
   // vae_decoder is destroyed here
 
   printf("\nOutput saved to %s (%dx%d)\n", FLAGS_output.c_str(), width, height);
   printf("Convert to PNG:  convert %s output.png\n", FLAGS_output.c_str());
   fflush(stdout);
 
   return 0;
 }
 