#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.


# Root Directories
GPUS="8" # GPU size for tensor_parallel.
ROOT_DIR="benchmark_root"
MODEL_DIR=""
ENGINE_DIR="."
BATCH_SIZE=1

# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
# Benchmark and Tasks
source config_tasks.sh
BENCHMARK="synthetic"

synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multikey_3"
    "niah_multivalue"
    "niah_multiquery"
    "vt"
    "cwe"
    "fwe"
    "qa_1"
    "qa_2"
)
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi


# Start client (prepare data / call model API / obtain final metrics)
total_time=0
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"

    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    
    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done

echo "Total time spent on call_api: $total_time seconds"
