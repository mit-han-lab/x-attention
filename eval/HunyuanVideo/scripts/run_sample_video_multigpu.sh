#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# Supported Parallel Configurations
# |     --video-size     | --video-length | --ulysses-degree x --ring-degree | --nproc_per_node |
# |----------------------|----------------|----------------------------------|------------------|
# | 1280 720 or 720 1280 | 129            | 8x1,4x2,2x4,1x8                  | 8                |
# | 1280 720 or 720 1280 | 129            | 1x5                              | 5                |
# | 1280 720 or 720 1280 | 129            | 4x1,2x2,1x4                      | 4                |
# | 1280 720 or 720 1280 | 129            | 3x1,1x3                          | 3                |
# | 1280 720 or 720 1280 | 129            | 2x1,1x2                          | 2                |
# | 1104 832 or 832 1104 | 129            | 4x1,2x2,1x4                      | 4                |
# | 1104 832 or 832 1104 | 129            | 3x1,1x3                          | 3                |
# | 1104 832 or 832 1104 | 129            | 2x1,1x2                          | 2                |
# | 960 960              | 129            | 6x1,3x2,2x3,1x6                  | 6                |
# | 960 960              | 129            | 4x1,2x2,1x4                      | 4                |
# | 960 960              | 129            | 3x1,1x3                          | 3                |
# | 960 960              | 129            | 1x2,2x1                          | 2                |
# | 960 544 or 544 960   | 129            | 6x1,3x2,2x3,1x6                  | 6                |
# | 960 544 or 544 960   | 129            | 4x1,2x2,1x4                      | 4                |
# | 960 544 or 544 960   | 129            | 3x1,1x3                          | 3                |
# | 960 544 or 544 960   | 129            | 1x2,2x1                          | 2                |
# | 832 624 or 624 832   | 129            | 4x1,2x2,1x4                      | 4                |
# | 624 832 or 624 832   | 129            | 3x1,1x3                          | 3                |
# | 832 624 or 624 832   | 129            | 2x1,1x2                          | 2                |
# | 720 720              | 129            | 1x5                              | 5                |
# | 720 720              | 129            | 3x1,1x3                          | 3                |

export TOKENIZERS_PARALLELISM=false

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8
export RING_DEGREE=1

torchrun --nproc_per_node=$NPROC_PER_NODE sample_video.py \
	--video-size 720 1280 \
	--video-length 129 \
	--infer-steps 50 \
	--prompt "A cat walks on the grass, realistic style." \
	--seed 42 \
	--embedded-cfg-scale 6.0 \
	--flow-shift 7.0 \
	--flow-reverse \
	--ulysses-degree=$ULYSSES_DEGREE \
	--ring-degree=$RING_DEGREE \
	--save-path ./results
