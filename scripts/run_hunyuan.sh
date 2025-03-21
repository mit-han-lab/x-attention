cd eval/HunyuanVideo
# Xattention
python run_vbench.py --num_sampled_prompts 1 --attn xattn --threshold 0.95 --stride 8
# Full attention
python run_vbench.py --num_sampled_prompts 1 --attn flash
