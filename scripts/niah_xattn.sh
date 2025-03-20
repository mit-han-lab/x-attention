cd eval/niah
mkdir -p logs img results

model=$1
context_lengths_min=$2
s_len=$3
pretrained_len=$4
model_provider=$5

suffix="xattn-default"
(
    python -u needle_in_haystack.py --s_len $s_len \
        --e_len $pretrained_len \
        --context_lengths_min $context_lengths_min \
        --context_lengths_max $pretrained_len \
        --model_provider $model_provider \
        --model_name_suffix $suffix \
        --simulation_length 0 \
        --context_lengths_num_intervals 13 \
        --document_depth_percent_intervals 10 \
        --model_path ../../models/${model}
) 2>&1 | tee logs/eval_${model}_${suffix}.log

# python visualize.py \
#     --folder_path "results/${model}_${suffix}/" \
#     --model_name "${model} XAttn Default" \
#     --pretrained_len $pretrained_len
