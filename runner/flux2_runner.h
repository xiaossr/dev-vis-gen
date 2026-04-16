/*
 * FLUX.2-klein-4B ExecuTorch QNN runner for Qualcomm HTP/DSP.
 *
 * Loads 3 .pte models (text_encoder, transformer, vae_decoder) and runs
 * the full text-to-image diffusion pipeline on-device.
 *
 * Pipeline:
 *   tokenize(prompt) → text_encoder → [4-step flow-matching] transformer → unpack → vae → image
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

namespace flux2 {

struct Config {
  int height = 512;
  int width = 512;
  int max_text_len = 512;
  int num_inference_steps = 4;
  int in_channels = 128;          // transformer patch channels (2*2*latent_ch)
  int joint_attention_dim = 7680; // text encoder output dim
  int latent_channels = 32;       // VAE latent channels
  int vae_scale_factor = 8;
  // Derived
  int patch_h() const { return height / (vae_scale_factor * 2); } // 32
  int patch_w() const { return width / (vae_scale_factor * 2); }  // 32
  int num_patches() const { return patch_h() * patch_w(); }       // 1024
  int latent_h() const { return height / vae_scale_factor; }      // 64
  int latent_w() const { return width / vae_scale_factor; }       // 64
};

class Runner {
 public:
  explicit Runner(
      const std::string& text_encoder_path,
      const std::string& transformer_path,
      const std::string& vae_path,
      const Config& config);

  struct Stats {
    static constexpr long SCALING_FACTOR_UNITS_PER_SECOND = 1000;
    long model_load_start_ms = 0;
    long model_load_end_ms = 0;
    long text_encoder_ms = 0;
    long transformer_total_ms = 0;
    long vae_ms = 0;
    long total_generate_ms = 0;
  };

  // Load all 3 models into memory
  executorch::runtime::Error load();
  bool is_loaded() const;

  // Run full pipeline: prompt → image (NCHW float32 in [0,1])
  executorch::runtime::Error generate(
      const std::vector<int64_t>& input_ids,
      const std::vector<int64_t>& attention_mask,
      std::vector<float>& output_image);

  // Save image as PPM file
  static void save_ppm(
      const std::vector<float>& image,
      int height,
      int width,
      const std::string& path);

  const Stats& stats() const { return stats_; }
  void print_performance() const;

 private:
  // Flow-matching schedule: linear sigma from 1→0
  std::vector<float> get_sigmas() const;

  // Build position IDs for transformer
  void build_img_ids(std::vector<float>& img_ids) const;
  void build_txt_ids(std::vector<float>& txt_ids) const;

  // Unpack transformer output [1,N,C] → VAE input [1,Ch,H,W]
  void unpack_latents(
      const std::vector<float>& packed,
      std::vector<float>& unpacked) const;

  Config config_;
  Stats stats_;
  std::unique_ptr<executorch::extension::Module> text_encoder_;
  std::unique_ptr<executorch::extension::Module> transformer_;
  std::unique_ptr<executorch::extension::Module> vae_;
  bool loaded_ = false;
};

} // namespace flux2
