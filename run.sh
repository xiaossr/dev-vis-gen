adb shell "cd /data/local/tmp/flux2 && \
    chmod +x flux2_main && \
    ./flux2_main \
        --model_dir . \
        --tokens prompt.bin \
        --output output.ppm \
        --steps 4 \
        --seed 42"