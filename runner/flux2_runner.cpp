/*
 * FLUX.2-klein-4B ExecuTorch QNN runner implementation.
 *
 * Flow-matching diffusion with 4-step Euler schedule (distilled, no CFG).
 */

#include "flux2_runner.h"

#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <random>

using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;

namespace flux2 {

static long time_in_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

Runner::Runner(
    const std::string& text_encoder_path,
    const std::string& transformer_path,
    const std::string& vae_path,
    const Config& config)
    : config_(config) {
  text_encoder_ = std::make_unique<Module>(
      text_encoder_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  transformer_ = std::make_unique<Module>(
      transformer_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
  vae_ = std::make_unique<Module>(
      vae_path, Module::LoadMode::MmapUseMlockIgnoreErrors);

  ET_LOG(Info, "Created FLUX.2 runner");
  ET_LOG(Info, "  text_encoder: %s", text_encoder_path.c_str());
  ET_LOG(Info, "  transformer:  %s", transformer_path.c_str());
  ET_LOG(Info, "  vae_decoder:  %s", vae_path.c_str());
  ET_LOG(
      Info,
      "  resolution: %dx%d, steps: %d, patches: %d",
      config_.height,
      config_.width,
      config_.num_inference_steps,
      config_.num_patches());
}

bool Runner::is_loaded() const {
  return loaded_;
}

Error Runner::load() {
  if (loaded_)
    return Error::Ok;

  stats_.model_load_start_ms = time_in_ms();

  // Load each model's forward method
  auto load_module = [](Module& mod, const char* name) -> Error {
    auto method_names = mod.method_names();
    if (!method_names.ok() || method_names->empty()) {
      ET_LOG(Error, "No methods found in %s", name);
      return Error::InvalidProgram;
    }
    const std::string& method = *method_names->begin();
    ET_LOG(Info, "Loading %s method: %s", name, method.c_str());
    return mod.load_method(method);
  };

  ET_CHECK_OK_OR_RETURN_ERROR(load_module(*text_encoder_, "text_encoder"));
  ET_CHECK_OK_OR_RETURN_ERROR(load_module(*transformer_, "transformer"));
  ET_CHECK_OK_OR_RETURN_ERROR(load_module(*vae_, "vae_decoder"));

  stats_.model_load_end_ms = time_in_ms();
  loaded_ = true;
  ET_LOG(
      Info,
      "All models loaded in %.1f seconds",
      (stats_.model_load_end_ms - stats_.model_load_start_ms) / 1000.0);
  return Error::Ok;
}

// Linear schedule: sigmas go from 1.0 → 0.0 in (num_steps + 1) points
std::vector<float> Runner::get_sigmas() const {
  int n = config_.num_inference_steps;
  std::vector<float> sigmas(n + 1);
  for (int i = 0; i <= n; ++i) {
    sigmas[i] = 1.0f - static_cast<float>(i) / static_cast<float>(n);
  }
  return sigmas; // [1.0, 0.75, 0.5, 0.25, 0.0] for 4 steps
}

// img_ids: [1, num_patches, 4] — (time=0, h, w, layer=0) for each patch
void Runner::build_img_ids(std::vector<float>& img_ids) const {
  int ph = config_.patch_h();
  int pw = config_.patch_w();
  int N = ph * pw;
  img_ids.resize(1 * N * 4, 0.0f);

  for (int h = 0; h < ph; ++h) {
    for (int w = 0; w < pw; ++w) {
      int idx = (h * pw + w) * 4;
      img_ids[idx + 0] = 0.0f;                    // time
      img_ids[idx + 1] = static_cast<float>(h);   // h
      img_ids[idx + 2] = static_cast<float>(w);   // w
      img_ids[idx + 3] = 0.0f;                    // layer
    }
  }
}

// txt_ids: [1, max_text_len, 4] — (time=0, h=0, w=0, seq) for each token
void Runner::build_txt_ids(std::vector<float>& txt_ids) const {
  int S = config_.max_text_len;
  txt_ids.resize(1 * S * 4, 0.0f);

  for (int s = 0; s < S; ++s) {
    int idx = s * 4;
    txt_ids[idx + 0] = 0.0f;                    // time
    txt_ids[idx + 1] = 0.0f;                    // h
    txt_ids[idx + 2] = 0.0f;                    // w
    txt_ids[idx + 3] = static_cast<float>(s);   // seq
  }
}

// Unpack: [1, patch_h*patch_w, 2*2*latent_ch] → [1, latent_ch, latent_h, latent_w]
// The packing is: for each (ph, pw), the 128 channels encode a 2x2 spatial patch
// of 32 latent channels: layout is [ph, pw, 2, 2, 32] → flatten last 3 dims = 128
void Runner::unpack_latents(
    const std::vector<float>& packed,
    std::vector<float>& unpacked) const {
  int ph = config_.patch_h();    // 32
  int pw = config_.patch_w();    // 32
  int C = config_.latent_channels; // 32
  int lh = config_.latent_h();   // 64
  int lw = config_.latent_w();   // 64

  unpacked.resize(1 * C * lh * lw);

  // packed shape: [1, ph*pw, 2*2*C]
  // For patch (i, j), the 128 values are arranged as: [dy=0,dx=0,c=0..31], [dy=0,dx=1,c=0..31], [dy=1,dx=0,...], [dy=1,dx=1,...]
  for (int i = 0; i < ph; ++i) {
    for (int j = 0; j < pw; ++j) {
      int patch_idx = i * pw + j;
      const float* src = packed.data() + patch_idx * (2 * 2 * C);

      for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
          int h_out = i * 2 + dy;
          int w_out = j * 2 + dx;
          for (int c = 0; c < C; ++c) {
            // Output NCHW: [0, c, h_out, w_out]
            int out_idx = c * lh * lw + h_out * lw + w_out;
            // Input: [patch_idx, dy*2*C + dx*C + c]
            int in_idx = (dy * 2 + dx) * C + c;
            unpacked[out_idx] = src[in_idx];
          }
        }
      }
    }
  }
}

Error Runner::generate(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask,
    std::vector<float>& output_image) {
  if (!loaded_) {
    ET_LOG(Error, "Models not loaded");
    return Error::InvalidState;
  }

  long gen_start = time_in_ms();
  int S = config_.max_text_len;
  int N = config_.num_patches();
  int C_in = config_.in_channels;
  int C_text = config_.joint_attention_dim;

  // ── 1. Text Encoder ──────────────────────────────────────────────────
  ET_LOG(Info, "Running text encoder...");
  long te_start = time_in_ms();

  // Copy to non-const buffers for from_blob
  std::vector<int64_t> ids_buf(input_ids);
  std::vector<int64_t> mask_buf(attention_mask);

  auto ids_tensor = from_blob(
      ids_buf.data(), {1, S}, executorch::aten::ScalarType::Long);
  auto mask_tensor = from_blob(
      mask_buf.data(), {1, S}, executorch::aten::ScalarType::Long);

  auto te_result = text_encoder_->forward({ids_tensor, mask_tensor});
  if (!te_result.ok()) {
    ET_LOG(Error, "Text encoder forward failed");
    return Error::InvalidState;
  }

  // Output: [1, S, joint_attention_dim]
  auto& te_outputs = te_result.get();
  if (te_outputs.empty()) {
    ET_LOG(Error, "Text encoder returned no outputs");
    return Error::InvalidState;
  }

  stats_.text_encoder_ms = time_in_ms() - te_start;
  ET_LOG(Info, "Text encoder done (%.1f s)", stats_.text_encoder_ms / 1000.0);

  // Get encoder_hidden_states tensor from output
  auto enc_hs_evalue = te_outputs[0];

  // ── 2. Prepare transformer inputs ────────────────────────────────────
  // Generate random initial latents [1, N, C_in]
  std::vector<float> latents(1 * N * C_in);
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float& v : latents) {
      v = dist(gen);
    }
  }

  // Build position IDs
  std::vector<float> img_ids_data, txt_ids_data;
  build_img_ids(img_ids_data);
  build_txt_ids(txt_ids_data);

  auto img_ids_tensor = from_blob(
      img_ids_data.data(), {1, N, 4}, executorch::aten::ScalarType::Float);
  auto txt_ids_tensor = from_blob(
      txt_ids_data.data(), {1, S, 4}, executorch::aten::ScalarType::Float);

  // Sigma schedule
  std::vector<float> sigmas = get_sigmas();

  // ── 3. Denoising loop (flow-matching Euler) ──────────────────────────
  ET_LOG(Info, "Running %d denoising steps...", config_.num_inference_steps);
  long tf_total_start = time_in_ms();

  for (int step = 0; step < config_.num_inference_steps; ++step) {
    float sigma = sigmas[step];
    float sigma_next = sigmas[step + 1];
    float dt = sigma_next - sigma; // negative (going 1→0)

    float timestep_val = sigma; // FLUX uses sigma as timestep directly
    std::vector<float> ts_buf = {timestep_val};

    auto latent_tensor = from_blob(
        latents.data(), {1, N, C_in}, executorch::aten::ScalarType::Float);
    auto timestep_tensor = from_blob(
        ts_buf.data(), {1}, executorch::aten::ScalarType::Float);

    long step_start = time_in_ms();

    // transformer(hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids)
    auto tf_result = transformer_->forward(
        {latent_tensor,
         enc_hs_evalue,
         timestep_tensor,
         img_ids_tensor,
         txt_ids_tensor});

    if (!tf_result.ok()) {
      ET_LOG(Error, "Transformer step %d failed", step);
      return Error::InvalidState;
    }

    long step_ms = time_in_ms() - step_start;
    ET_LOG(Info, "  Step %d/%d (sigma=%.3f, dt=%.3f) — %.1f s",
           step + 1, config_.num_inference_steps, sigma, dt, step_ms / 1000.0);

    // Get velocity prediction
    auto& tf_outputs = tf_result.get();
    const auto& velocity_tensor = tf_outputs[0].toTensor();
    const float* velocity = velocity_tensor.const_data_ptr<float>();

    // Euler step: latents = latents + velocity * dt
    for (int i = 0; i < static_cast<int>(latents.size()); ++i) {
      latents[i] = latents[i] + velocity[i] * dt;
    }
  }

  stats_.transformer_total_ms = time_in_ms() - tf_total_start;
  ET_LOG(
      Info,
      "Denoising done (%.1f s total, %.1f s/step avg)",
      stats_.transformer_total_ms / 1000.0,
      stats_.transformer_total_ms / 1000.0 / config_.num_inference_steps);

  // ── 4. Unpack latents ────────────────────────────────────────────────
  ET_LOG(Info, "Unpacking latents...");
  int C_lat = config_.latent_channels;
  int lh = config_.latent_h();
  int lw = config_.latent_w();
  std::vector<float> vae_input;
  unpack_latents(latents, vae_input);

  // ── 5. VAE Decode ────────────────────────────────────────────────────
  ET_LOG(Info, "Running VAE decoder...");
  long vae_start = time_in_ms();

  auto vae_input_tensor = from_blob(
      vae_input.data(),
      {1, C_lat, lh, lw},
      executorch::aten::ScalarType::Float);

  auto vae_result = vae_->forward({vae_input_tensor});
  if (!vae_result.ok()) {
    ET_LOG(Error, "VAE decoder failed");
    return Error::InvalidState;
  }

  stats_.vae_ms = time_in_ms() - vae_start;
  ET_LOG(Info, "VAE done (%.1f s)", stats_.vae_ms / 1000.0);

  // Copy output image [1, 3, H, W]
  auto& vae_outputs = vae_result.get();
  const auto& img_tensor = vae_outputs[0].toTensor();
  const float* img_data = img_tensor.const_data_ptr<float>();
  int img_size = 1 * 3 * config_.height * config_.width;
  output_image.assign(img_data, img_data + img_size);

  stats_.total_generate_ms = time_in_ms() - gen_start;
  return Error::Ok;
}

void Runner::save_ppm(
    const std::vector<float>& image,
    int height,
    int width,
    const std::string& path) {
  // Image is NCHW float32, values may be outside [0,1]
  // PPM format: P6, width height, max_val, then RGB bytes
  std::ofstream f(path, std::ios::binary);
  f << "P6\n" << width << " " << height << "\n255\n";

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        // NCHW: index = c * H * W + y * W + x
        float v = image[c * height * width + y * width + x];
        // Clamp to [0, 1] and convert to byte
        v = std::max(0.0f, std::min(1.0f, (v + 1.0f) / 2.0f)); // [-1,1] → [0,1]
        uint8_t b = static_cast<uint8_t>(v * 255.0f);
        f.put(static_cast<char>(b));
      }
    }
  }
  f.close();
}

void Runner::print_performance() const {
  ET_LOG(Info, "=== FLUX.2 Performance ===");
  ET_LOG(
      Info,
      "  Model load:       %.1f s",
      (stats_.model_load_end_ms - stats_.model_load_start_ms) / 1000.0);
  ET_LOG(Info, "  Text encoder:     %.1f s", stats_.text_encoder_ms / 1000.0);
  ET_LOG(
      Info,
      "  Transformer:      %.1f s total (%.1f s/step x %d steps)",
      stats_.transformer_total_ms / 1000.0,
      stats_.transformer_total_ms / 1000.0 / config_.num_inference_steps,
      config_.num_inference_steps);
  ET_LOG(Info, "  VAE decoder:      %.1f s", stats_.vae_ms / 1000.0);
  ET_LOG(
      Info, "  Total generation: %.1f s", stats_.total_generate_ms / 1000.0);
}

} // namespace flux2
