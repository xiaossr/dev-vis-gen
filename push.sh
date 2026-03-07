adb shell mkdir -p /data/local/tmp/flux2

# Push .pte models
adb push exported_flux2_klein/text_encoder.pte /data/local/tmp/flux2/
adb push exported_flux2_klein/transformer.pte  /data/local/tmp/flux2/
adb push exported_flux2_klein/vae_decoder.pte  /data/local/tmp/flux2/

# Push binary inputs
adb push exported_flux2_klein/prompt.bin  /data/local/tmp/flux2/
adb push exported_flux2_klein/bn_mean.bin /data/local/tmp/flux2/
adb push exported_flux2_klein/bn_var.bin  /data/local/tmp/flux2/

# Push runner binary
adb push ../executorch/cmake-out-android/examples/models/flux2/flux2_main /data/local/tmp/flux2/