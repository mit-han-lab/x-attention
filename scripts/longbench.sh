model=$1
task=$2
method=$3
python -u eval/LongBench/pred.py \
    --model $model --task $task \
    --method $method
