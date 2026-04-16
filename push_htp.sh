#!/bin/bash
# Push FLUX.2-klein HTP files to Android device
# Text encoder: XNNPACK CPU
# Transformer: QNN HTP quantized
# VAE decoder: QNN HTP fp16

SRC=/Users/aprilhu/Desktop/vm_shared/dev-vis-gen/exported_flux2_klein
DST=/data/local/tmp/flux2/htp

# adb shell mkdir -p $DST

# .pte models
adb push $SRC/text_encoder.pte $DST/
adb push $SRC/transformer.pte $DST/
adb push $SRC/vae_decoder.pte $DST/

# Binary inputs
adb push $SRC/prompt.bin $DST/
adb push $SRC/bn_mean.bin $DST/
adb push $SRC/bn_var.bin $DST/

# Runner binary
adb push $SRC/flux2_qnn_main $DST/
adb shell chmod +x $DST/flux2_qnn_main

# QNN HTP libraries (adjust paths to your QNN SDK)
# adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so $DST/
# adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so $DST/
# adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so $DST/
# adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so $DST/
# adb push libqnn_executorch_backend.so $DST/

echo "Done. Run with:"
echo "adb shell \"cd $DST && export LD_LIBRARY_PATH=. && ./flux2_qnn_main --model_dir . --tokens prompt.bin --output output.ppm --steps 4 --seed 42 --htp_performance_mode 3\""
