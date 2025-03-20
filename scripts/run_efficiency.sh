rm -rf output/*
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval/efficiency/attention_speedup.py