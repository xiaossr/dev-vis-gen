/*
 * FLUX.2-klein-4B on-device inference entry point.
 *
 * Usage:
 *   ./flux2_runner \
 *     --model_dir ./exported_flux2_klein_qnn \
 *     --prompt "a cat sitting on a windowsill at sunset" \
 *     --output output.ppm
 *
 * The model_dir should contain:
 *   text_encoder.pte, transformer.pte, vae_decoder.pte,
 *   tokenizer/ (with tokenizer.json), export_config.json
 */

#include "flux2_runner.h"

#include <executorch/runtime/platform/runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Simple argument parsing (no gflags dependency for portability)
struct Args {
  std::string model_dir = "./exported_flux2_klein_qnn";
  std::string prompt = "a photograph of an astronaut riding a horse";
  std::string output = "output.ppm";
  int num_steps = 4;
  int seed = -1; // -1 = random
};

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--model_dir" || arg == "-d") && i + 1 < argc)
      args.model_dir = argv[++i];
    else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc)
      args.prompt = argv[++i];
    else if ((arg == "--output" || arg == "-o") && i + 1 < argc)
      args.output = argv[++i];
    else if (arg == "--steps" && i + 1 < argc)
      args.num_steps = std::atoi(argv[++i]);
    else if (arg == "--seed" && i + 1 < argc)
      args.seed = std::atoi(argv[++i]);
    else if (arg == "--help" || arg == "-h") {
      std::cout
          << "FLUX.2-klein-4B on-device inference\n"
          << "Usage: " << argv[0] << " [options]\n"
          << "  --model_dir DIR   Directory with .pte files (default: ./exported_flux2_klein_qnn)\n"
          << "  --prompt TEXT     Text prompt (default: astronaut on horse)\n"
          << "  --output FILE     Output PPM path (default: output.ppm)\n"
          << "  --steps N         Denoising steps (default: 4)\n"
          << "  --seed N          Random seed (-1 = random)\n";
      std::exit(0);
    }
  }
  return args;
}

// Minimal tokenization placeholder.
// In production, use the saved Qwen3 tokenizer (tokenizer.json via sentencepiece or tiktoken).
// This creates simple input_ids/attention_mask for testing the pipeline.
static void simple_tokenize(
    const std::string& prompt,
    int max_len,
    std::vector<int64_t>& input_ids,
    std::vector<int64_t>& attention_mask) {
  // TODO: Replace with proper Qwen3 BPE tokenizer.
  // For now, use byte-level encoding as a placeholder to test the pipeline.
  input_ids.clear();
  attention_mask.clear();

  // BOS token (Qwen3 typically uses 151643 for <|im_start|> or similar)
  input_ids.push_back(1);
  for (char c : prompt) {
    input_ids.push_back(static_cast<int64_t>(static_cast<unsigned char>(c)));
  }
  // EOS
  input_ids.push_back(2);

  // Pad to max_len
  int real_len = static_cast<int>(input_ids.size());
  input_ids.resize(max_len, 0);
  attention_mask.resize(max_len, 0);
  for (int i = 0; i < real_len && i < max_len; ++i) {
    attention_mask[i] = 1;
  }
}

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();

  Args args = parse_args(argc, argv);

  std::cout << "FLUX.2-klein-4B Inference Runner\n";
  std::cout << "  Model dir: " << args.model_dir << "\n";
  std::cout << "  Prompt:    " << args.prompt << "\n";
  std::cout << "  Steps:     " << args.num_steps << "\n";
  std::cout << "  Output:    " << args.output << "\n\n";

  // Configure
  flux2::Config config;
  config.num_inference_steps = args.num_steps;
  // Other config values match export_config.json defaults

  // Create runner
  flux2::Runner runner(
      args.model_dir + "/text_encoder.pte",
      args.model_dir + "/transformer.pte",
      args.model_dir + "/vae_decoder.pte",
      config);

  // Load models
  std::cout << "Loading models...\n";
  auto err = runner.load();
  if (err != executorch::runtime::Error::Ok) {
    std::cerr << "Failed to load models\n";
    return 1;
  }

  // Tokenize
  std::vector<int64_t> input_ids, attention_mask;
  simple_tokenize(args.prompt, config.max_text_len, input_ids, attention_mask);
  std::cout << "Tokenized prompt (" << input_ids.size() << " tokens)\n";

  // Generate
  std::cout << "Generating image...\n";
  std::vector<float> image;
  err = runner.generate(input_ids, attention_mask, image);
  if (err != executorch::runtime::Error::Ok) {
    std::cerr << "Generation failed\n";
    return 1;
  }

  // Save
  flux2::Runner::save_ppm(image, config.height, config.width, args.output);
  std::cout << "Saved " << args.output << " (" << config.height << "x"
            << config.width << ")\n";

  runner.print_performance();
  return 0;
}
