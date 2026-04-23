#!/bin/bash
# Push FLUX.2-klein-4B QNN HTP artefacts + runtime to Android device.
#
# Assumes:
#   - You ran export_flux2_klein_qnn.py with --soc_model SM8750 (or $SOC)
#   - ExecuTorch Android build with EXECUTORCH_BUILD_QNN=ON exists
#   - runner/build-android/flux2_runner was built with ENABLE_QNN=ON
#   - QAIRT SDK is at $QAIRT (default below)
#
# Usage: ./push_htp.sh [src_dir] [soc]
set -euo pipefail

REPO=/data/home/thanush/dev-vis-gen
SRC="${1:-$REPO/exported_flux2_klein_qnn_smoketest}"
SOC="${2:-SM8750}"
QAIRT="${QAIRT:-$REPO/qairt/2.45.0.260326}"
DST=/data/local/tmp/flux2/htp

# SoC → Hexagon arch
declare -A HEXARCH=( [SM8750]=v79 [SM8650]=v75 [SM8550]=v73 [SM8450]=v69 )
HEX="${HEXARCH[$SOC]:-v79}"

echo "Target SoC: $SOC  (Hexagon $HEX)"
echo "Source    : $SRC"

adb shell mkdir -p "$DST"

# .pte models
for f in text_encoder.pte transformer.pte vae_decoder.pte; do
  [ -f "$SRC/$f" ] && adb push "$SRC/$f" "$DST/"
done

# Metadata + tokenizer + BN stats
adb push "$SRC/export_config.json" "$DST/"
[ -f "$SRC/vae_bn_stats.pt" ] && adb push "$SRC/vae_bn_stats.pt" "$DST/"
[ -d "$SRC/tokenizer" ]      && adb push "$SRC/tokenizer" "$DST/"

# Binary inputs (from prepare_mobile.py)
for b in prompt.bin bn_mean.bin bn_var.bin; do
  [ -f "$SRC/$b" ] && adb push "$SRC/$b" "$DST/"
done

# Runner binary (built with ENABLE_QNN=ON)
RUNNER=$REPO/runner/build-android/flux2_runner
adb push "$RUNNER" "$DST/"
adb shell chmod +x "$DST/flux2_runner"

# ExecuTorch QNN backend shared lib
adb push "$REPO/executorch/build-android/backends/qualcomm/libqnn_executorch_backend.so" "$DST/"

# QAIRT Android-side runtime (ARM64)
AND=$QAIRT/lib/aarch64-android
adb push "$AND/libQnnHtp.so"        "$DST/"
adb push "$AND/libQnnSystem.so"     "$DST/"
adb push "$AND/libQnnHtpPrepare.so" "$DST/"
adb push "$AND/libQnnHtp${HEX^^}Stub.so" "$DST/"

# QAIRT Hexagon DSP-side skeleton (runs on DSP itself)
DSP=$QAIRT/lib/hexagon-$HEX/unsigned
adb push "$DSP/libQnnHtp${HEX^^}Skel.so" "$DST/"

echo
echo "Done. Run on device with:"
echo "  adb shell \"cd $DST && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./flux2_runner --model_dir . --tokens prompt.bin --output output.ppm --steps 4 --seed 42\""
