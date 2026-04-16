#!/bin/bash
# Deploy FLUX.2-klein-4B to a Qualcomm Snapdragon device.
#
# Prerequisites:
#   1. Android NDK installed (r25c+ recommended)
#   2. QNN SDK installed ($QNN_SDK_ROOT)
#   3. ExecuTorch built for Android ARM64 with QNN backend
#   4. Device connected via USB with adb
#
# Usage:
#   export ANDROID_NDK=/path/to/android-ndk-r25c
#   export QNN_SDK_ROOT=/path/to/qairt/2.45.0.260326
#   export EXECUTORCH_ROOT=/path/to/executorch
#   ./deploy_to_device.sh [--build-only] [--push-only] [--run-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SCRIPT_DIR}/build-android"
DEVICE_DIR="/data/local/tmp/flux2"
MODEL_DIR="${PROJECT_DIR}/exported_flux2_klein_qnn"

# Defaults
ANDROID_NDK="${ANDROID_NDK:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-}"
EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-${PROJECT_DIR}/executorch}"

check_prereqs() {
  if [ -z "$ANDROID_NDK" ]; then
    echo "ERROR: Set ANDROID_NDK to your Android NDK path"
    exit 1
  fi
  if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: Set QNN_SDK_ROOT to your QNN SDK path"
    exit 1
  fi
  if ! command -v adb &>/dev/null; then
    echo "ERROR: adb not found in PATH"
    exit 1
  fi
}

build() {
  echo "=== Building FLUX.2 runner for Android ARM64 ==="
  check_prereqs

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake "$SCRIPT_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-30 \
    -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
    -DQNN_SDK_ROOT="${QNN_SDK_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release

  cmake --build . -j"$(nproc)"
  echo "Build complete: ${BUILD_DIR}/flux2_runner"
}

push() {
  echo "=== Pushing files to device ==="
  check_prereqs

  adb shell "mkdir -p ${DEVICE_DIR}"

  # Push runner binary
  echo "Pushing runner binary..."
  adb push "${BUILD_DIR}/flux2_runner" "${DEVICE_DIR}/"

  # Push .pte model files (~6.7 GB total, this will take a while)
  echo "Pushing model files (this may take 10+ minutes)..."
  for f in text_encoder.pte transformer.pte vae_decoder.pte export_config.json; do
    if [ -f "${MODEL_DIR}/${f}" ]; then
      echo "  ${f} ($(du -h "${MODEL_DIR}/${f}" | cut -f1))"
      adb push "${MODEL_DIR}/${f}" "${DEVICE_DIR}/"
    fi
  done

  # Push tokenizer
  adb shell "mkdir -p ${DEVICE_DIR}/tokenizer"
  adb push "${MODEL_DIR}/tokenizer/" "${DEVICE_DIR}/tokenizer/"

  # Push QNN HTP runtime libraries
  QNN_LIB="${QNN_SDK_ROOT}/lib/aarch64-android"
  if [ -d "$QNN_LIB" ]; then
    echo "Pushing QNN runtime libraries..."
    for lib in libQnnHtp.so libQnnHtpV75Stub.so libQnnHtpV75Skel.so \
               libQnnSystem.so libQnnHtpPrepare.so libQnnHtpNetRunExtensions.so; do
      if [ -f "${QNN_LIB}/${lib}" ]; then
        adb push "${QNN_LIB}/${lib}" "${DEVICE_DIR}/"
      fi
    done
    # Also push hexagon skel libs if present
    HEX_LIB="${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned"
    if [ -d "$HEX_LIB" ]; then
      for lib in "${HEX_LIB}"/*.so; do
        adb push "$lib" "${DEVICE_DIR}/"
      done
    fi
  else
    echo "WARNING: QNN aarch64-android libs not found at ${QNN_LIB}"
  fi

  adb shell "chmod +x ${DEVICE_DIR}/flux2_runner"
  echo "Push complete."
}

run() {
  echo "=== Running FLUX.2 on device ==="
  local prompt="${1:-a photograph of an astronaut riding a horse}"

  adb shell "cd ${DEVICE_DIR} && \
    export LD_LIBRARY_PATH=${DEVICE_DIR}:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH='${DEVICE_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && \
    ./flux2_runner \
      --model_dir ${DEVICE_DIR} \
      --prompt '${prompt}' \
      --output ${DEVICE_DIR}/output.ppm \
      --steps 4"

  # Pull output image
  adb pull "${DEVICE_DIR}/output.ppm" "./output.ppm"
  echo "Output saved to ./output.ppm"
}

# Parse command
case "${1:-all}" in
  --build-only) build ;;
  --push-only) push ;;
  --run-only) shift; run "$@" ;;
  all|"")
    build
    push
    run "${2:-}"
    ;;
  *)
    echo "Usage: $0 [--build-only|--push-only|--run-only [prompt]]"
    exit 1
    ;;
esac
