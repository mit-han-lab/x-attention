model="Llama-3.1-8B-Instruct"
model_provider=LLaMA
context_lengths_min=8000
pretrained_len=128000

CUDA_VISIBLE_DEVICES=0,1 bash scripts/niah_xattn.sh $model $context_lengths_min 0 $pretrained_len $model_provider
