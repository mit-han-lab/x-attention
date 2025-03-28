#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model
DIT_CKPT_PATH={PATH_TO}/{MODEL_NAME}_model_states_fp8.pt

python3 sample_video.py \
    --dit-weight ${DIT_CKPT_PATH} \
    --video-size 720 1280 \
    --video-length 129 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./results
