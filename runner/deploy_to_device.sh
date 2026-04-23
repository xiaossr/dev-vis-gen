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

# Defaults
ANDROID_NDK="${ANDROID_NDK:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-${PROJECT_DIR}/qairt/2.45.0.260326}"
EXECUTORCH_ROOT="${EXECUTORCH_ROOT:-${PROJECT_DIR}/executorch}"
EXECUTORCH_INSTALL="${EXECUTORCH_INSTALL:-${EXECUTORCH_ROOT}/install-android}"
MOBILE_PROMPT="${MOBILE_PROMPT:-a photograph of an astronaut riding a horse}"
ADB_BIN="${ADB:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

pick_default_model_dir() {
  local candidate
  for candidate in \
    "${PROJECT_DIR}/exported_flux2_klein_qnn_v81" \
    "${PROJECT_DIR}/exported_flux2_klein_qnn_full" \
    "${PROJECT_DIR}/exported_flux2_klein_qnn" \
    "${PROJECT_DIR}/exported_flux2_klein_xnnpack"
  do
    if [ -f "${candidate}/export_config.json" ]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "${PROJECT_DIR}/exported_flux2_klein_qnn_v81"
}

MODEL_DIR="${MODEL_DIR:-$(pick_default_model_dir)}"

resolve_adb() {
  if [ -n "${ADB_BIN}" ]; then
    echo "${ADB_BIN}"
    return 0
  fi
  if command -v adb &>/dev/null; then
    command -v adb
    return 0
  fi
  if [ -x "${PROJECT_DIR}/.tools/platform-tools/adb" ]; then
    echo "${PROJECT_DIR}/.tools/platform-tools/adb"
    return 0
  fi
  return 1
}

resolve_python() {
  if [ -n "${PYTHON_BIN}" ]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  if [ -x "${PROJECT_DIR}/.venv/bin/python" ]; then
    echo "${PROJECT_DIR}/.venv/bin/python"
    return 0
  fi
  if command -v python3 &>/dev/null; then
    command -v python3
    return 0
  fi
  if command -v python &>/dev/null; then
    command -v python
    return 0
  fi
  return 1
}

ADB_BIN="$(resolve_adb || true)"
PYTHON_BIN="$(resolve_python || true)"

detect_qnn_arch() {
  local export_config="${MODEL_DIR}/export_config.json"
  if [ -f "$export_config" ]; then
    local soc
    if [ -z "${PYTHON_BIN}" ]; then
      echo "ERROR: python not found; set PYTHON_BIN or install a python interpreter"
      exit 1
    fi
    soc="$("${PYTHON_BIN}" - <<'PY' "$export_config"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("soc_model", ""))
PY
)"
    case "$soc" in
      SM8850) echo "v81" ; return 0 ;;
      SM8750) echo "v79" ; return 0 ;;
      SM8650) echo "v75" ; return 0 ;;
    esac
  fi
  echo "v75"
}

check_build_prereqs() {
  if [ -z "${ANDROID_NDK}" ]; then
    echo "ERROR: Set ANDROID_NDK to your Android NDK path"
    exit 1
  fi
}

check_adb() {
  if [ -z "${ADB_BIN}" ]; then
    echo "ERROR: adb not found in PATH and no repo-local platform-tools at ${PROJECT_DIR}/.tools/platform-tools/adb"
    exit 1
  fi
}

build() {
  echo "=== Building FLUX.2 runner for Android ARM64 ==="
  check_build_prereqs

  mkdir -p "$BUILD_DIR"
  cd "$BUILD_DIR"

  cmake "$SCRIPT_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-30 \
    -DCMAKE_BUILD_TYPE=Release \
    -Dexecutorch_DIR="${EXECUTORCH_INSTALL}/lib/cmake/ExecuTorch" \
    -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}"

  cmake --build . -j"$(nproc)"
  echo "Build complete: ${BUILD_DIR}/flux2_runner"
}

prep_inputs() {
  echo "=== Preparing host-side binary inputs ==="
  local prompt="${1:-a photograph of an astronaut riding a horse}"
  if [ -z "${PYTHON_BIN}" ]; then
    echo "ERROR: python not found; set PYTHON_BIN or install a python interpreter"
    exit 1
  fi
  (cd "${PROJECT_DIR}" && "${PYTHON_BIN}" prepare_mobile.py \
     --model_dir "${MODEL_DIR}" \
     --prompt "${prompt}" \
     --output_dir "${MODEL_DIR}")
}

ensure_mobile_inputs() {
  local f
  local missing=0
  for f in prompt.bin bn_mean.bin bn_var.bin; do
    if [ ! -f "${MODEL_DIR}/${f}" ]; then
      missing=1
      break
    fi
  done

  if [ "${missing}" -eq 0 ]; then
    return 0
  fi

  if [ ! -d "${MODEL_DIR}/tokenizer" ] || [ ! -f "${MODEL_DIR}/vae_bn_stats.pt" ]; then
    echo "ERROR: ${MODEL_DIR} is missing tokenizer/ or vae_bn_stats.pt; cannot generate mobile inputs"
    exit 1
  fi

  echo "Generating missing prompt.bin / BN inputs in ${MODEL_DIR}..."
  prep_inputs "${MOBILE_PROMPT}"
}

push() {
  echo "=== Pushing files to device ==="
  check_adb
  ensure_mobile_inputs

  "${ADB_BIN}" shell "mkdir -p ${DEVICE_DIR}"

  # Push runner binary
  echo "Pushing runner binary..."
  "${ADB_BIN}" push "${BUILD_DIR}/flux2_runner" "${DEVICE_DIR}/"

  if [ -f "${EXECUTORCH_INSTALL}/lib/libqnn_executorch_backend.so" ]; then
    echo "Pushing ExecuTorch QNN backend..."
    "${ADB_BIN}" push "${EXECUTORCH_INSTALL}/lib/libqnn_executorch_backend.so" "${DEVICE_DIR}/"
  fi

  # Push .pte model files (~5-8 GB total, this will take a while)
  echo "Pushing model files (this may take 10+ minutes)..."
  for f in text_encoder.pte transformer.pte vae_decoder.pte \
           export_config.json prompt.bin bn_mean.bin bn_var.bin vae_bn_stats.pt; do
    if [ -f "${MODEL_DIR}/${f}" ]; then
      echo "  ${f} ($(du -h "${MODEL_DIR}/${f}" | cut -f1))"
      "${ADB_BIN}" push "${MODEL_DIR}/${f}" "${DEVICE_DIR}/"
    fi
  done

  # Push QNN HTP runtime libraries (only if QNN_SDK_ROOT set — needed if
  # the .pte files embed a QNN delegate; for the all-XNNPACK XNNPACK builds
  # this can be skipped).
  if [ -n "${QNN_SDK_ROOT}" ]; then
    QNN_LIB="${QNN_SDK_ROOT}/lib/aarch64-android"
    local qnn_arch
    qnn_arch="$(detect_qnn_arch)"
    if [ -d "$QNN_LIB" ]; then
      echo "Pushing QNN runtime libraries for ${qnn_arch}..."
      for lib in libQnnHtp.so "libQnnHtp${qnn_arch^^}Stub.so" \
                 libQnnSystem.so libQnnHtpPrepare.so libQnnHtpNetRunExtensions.so; do
        if [ -f "${QNN_LIB}/${lib}" ]; then
          "${ADB_BIN}" push "${QNN_LIB}/${lib}" "${DEVICE_DIR}/"
        fi
      done
      HEX_LIB="${QNN_SDK_ROOT}/lib/hexagon-${qnn_arch}/unsigned"
      if [ -d "$HEX_LIB" ]; then
        for lib in "${HEX_LIB}"/*.so; do
          "${ADB_BIN}" push "$lib" "${DEVICE_DIR}/"
        done
      fi
    fi
  fi

  "${ADB_BIN}" shell "chmod +x ${DEVICE_DIR}/flux2_runner"
  echo "Push complete."
}

run() {
  echo "=== Running FLUX.2 on device ==="
  check_adb

  "${ADB_BIN}" shell "cd ${DEVICE_DIR} && \
    export LD_LIBRARY_PATH=${DEVICE_DIR}:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH='${DEVICE_DIR};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp' && \
    ./flux2_runner \
      --model_dir ${DEVICE_DIR} \
      --tokens ${DEVICE_DIR}/prompt.bin \
      --output ${DEVICE_DIR}/output.ppm \
      --steps 4"

  "${ADB_BIN}" pull "${DEVICE_DIR}/output.ppm" "./output.ppm"
  echo "Output saved to ./output.ppm"
}

# Parse command
case "${1:-all}" in
  --build-only) build ;;
  --push-only) push ;;
  --run-only) run ;;
  --prep-only) shift; prep_inputs "${1:-}" ;;
  all|"")
    build
    prep_inputs "${2:-}"
    push
    run
    ;;
  *)
    echo "Usage: $0 [--build-only|--push-only|--run-only|--prep-only [prompt]]"
    exit 1
    ;;
esac
