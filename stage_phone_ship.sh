#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
BUNDLE_DIR="${BUNDLE_DIR:-${PROJECT_DIR}/flux2_phone_ship}"
MODEL_DIR="${MODEL_DIR:-}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-${PROJECT_DIR}/qairt/2.45.0.260326}"
EXECUTORCH_INSTALL="${EXECUTORCH_INSTALL:-${PROJECT_DIR}/executorch/install-android}"
RUNNER_BIN="${RUNNER_BIN:-${PROJECT_DIR}/runner/build-android/flux2_runner}"
PROMPT="${PROMPT:-a photograph of an astronaut riding a horse}"
PYTHON_BIN="${PYTHON_BIN:-}"

pick_default_model_dir() {
  local candidate
  for candidate in \
    "${PROJECT_DIR}/exported_flux2_klein_qnn_v81" \
    "${PROJECT_DIR}/exported_flux2_klein_qnn_full" \
    "${PROJECT_DIR}/exported_flux2_klein_qnn"
  do
    if [ -f "${candidate}/export_config.json" ]; then
      echo "${candidate}"
      return 0
    fi
  done
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

copy_file() {
  local src="$1"
  local dst="$2"
  if [ ! -f "${src}" ]; then
    echo "ERROR: missing ${src}" >&2
    exit 1
  fi
  cp --reflink=auto --remove-destination "${src}" "${dst}"
}

copy_tree() {
  local src="$1"
  local dst="$2"
  if [ ! -d "${src}" ]; then
    echo "ERROR: missing ${src}" >&2
    exit 1
  fi
  rm -rf "${dst}"
  mkdir -p "${dst}"
  cp -a "${src}/." "${dst}/"
}

detect_qnn_arch() {
  local soc
  soc="$("${PYTHON_BIN}" - <<'PY' "${MODEL_DIR}/export_config.json"
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    print(json.load(f).get("soc_model", ""))
PY
)"
  case "${soc}" in
    SM8850) echo "v81" ;;
    SM8750) echo "v79" ;;
    SM8650) echo "v75" ;;
    *)
      echo "ERROR: unsupported soc_model '${soc}' in ${MODEL_DIR}/export_config.json" >&2
      exit 1
      ;;
  esac
}

MODEL_DIR="${MODEL_DIR:-$(pick_default_model_dir || true)}"
if [ -z "${MODEL_DIR}" ] || [ ! -f "${MODEL_DIR}/export_config.json" ]; then
  echo "ERROR: set MODEL_DIR to an exported QNN model directory with export_config.json" >&2
  exit 1
fi

PYTHON_BIN="$(resolve_python || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: python not found; set PYTHON_BIN or install a python interpreter" >&2
  exit 1
fi

QNN_ARCH="$(detect_qnn_arch)"
QNN_ARCH_UPPER="${QNN_ARCH^^}"
QNN_LIB_DIR="${QNN_SDK_ROOT}/lib/aarch64-android"
HEX_LIB_DIR="${QNN_SDK_ROOT}/lib/hexagon-${QNN_ARCH}/unsigned"

mkdir -p "${BUNDLE_DIR}"

echo "=== Staging FLUX.2 phone bundle ==="
echo "MODEL_DIR=${MODEL_DIR}"
echo "BUNDLE_DIR=${BUNDLE_DIR}"
echo "QNN_ARCH=${QNN_ARCH}"

copy_file "${MODEL_DIR}/text_encoder.pte" "${BUNDLE_DIR}/text_encoder.pte"
copy_file "${MODEL_DIR}/transformer.pte" "${BUNDLE_DIR}/transformer.pte"
copy_file "${MODEL_DIR}/vae_decoder.pte" "${BUNDLE_DIR}/vae_decoder.pte"
copy_file "${MODEL_DIR}/export_config.json" "${BUNDLE_DIR}/export_config.json"
copy_file "${MODEL_DIR}/vae_bn_stats.pt" "${BUNDLE_DIR}/vae_bn_stats.pt"
copy_tree "${MODEL_DIR}/tokenizer" "${BUNDLE_DIR}/tokenizer"

echo "Preparing prompt.bin / BN inputs..."
"${PYTHON_BIN}" "${PROJECT_DIR}/prepare_mobile.py" \
  --model_dir "${MODEL_DIR}" \
  --prompt "${PROMPT}" \
  --output_dir "${BUNDLE_DIR}"

copy_file "${RUNNER_BIN}" "${BUNDLE_DIR}/flux2_runner"
copy_file "${EXECUTORCH_INSTALL}/lib/libqnn_executorch_backend.so" \
  "${BUNDLE_DIR}/libqnn_executorch_backend.so"

copy_file "${QNN_LIB_DIR}/libQnnHtp.so" "${BUNDLE_DIR}/libQnnHtp.so"
copy_file "${QNN_LIB_DIR}/libQnnSystem.so" "${BUNDLE_DIR}/libQnnSystem.so"
copy_file "${QNN_LIB_DIR}/libQnnHtpPrepare.so" "${BUNDLE_DIR}/libQnnHtpPrepare.so"
copy_file "${QNN_LIB_DIR}/libQnnHtpNetRunExtensions.so" \
  "${BUNDLE_DIR}/libQnnHtpNetRunExtensions.so"
copy_file "${QNN_LIB_DIR}/libQnnHtp${QNN_ARCH_UPPER}Stub.so" \
  "${BUNDLE_DIR}/libQnnHtp${QNN_ARCH_UPPER}Stub.so"

for lib in "${HEX_LIB_DIR}"/*.so; do
  copy_file "${lib}" "${BUNDLE_DIR}/$(basename "${lib}")"
done

echo "Bundle staged at ${BUNDLE_DIR}"
