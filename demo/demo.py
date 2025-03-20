import time
import torch
from transformers import StaticCache
from xattn.src.load_llama import load_model,FastPrefillConfig
from eval.efficiency.generate_prompt import generate_prompt
import argparse
from tqdm import tqdm
if __name__ == "__main__":
    # load model and tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=450560)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default='xattn')
    args = parser.parse_args()
    config = FastPrefillConfig(metric = args.metric,stride = args.stride, threshold = args.threshold)
    
    model, tokenizer = load_model(name_or_path="gradientai/Llama-3-8B-Instruct-Gradient-1048k", fastprefillconfig=config)
    input_ids = generate_prompt(tokenizer,args.len)
    # -------------------
    # 1. Prefill
    # -------------------
    past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=450710, device=model.device, dtype=model.dtype)
    start_prefill = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, input_ids.size(1), args.chunk_size), desc="Prefilling", unit="chunk"):
            chunk = input_ids[:, i: i + args.chunk_size]
            output = model(
                input_ids=chunk,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = output.past_key_values
    torch.cuda.synchronize()
    end_prefill = time.time()
    prefill_time = end_prefill - start_prefill
    print(f"Prefill Time: {prefill_time:.4f} s")

     # -------------------
    # 2. Decode
    # -------------------
    start_decode = time.time()
    eos_token_id = tokenizer.eos_token_id  # 获取eos token ID
    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_content = [pred_token_idx.item()]
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in tqdm(range(50), desc="Decoding", unit="token"):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
                num_logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_token = pred_token_idx.item()
            
            if generated_token == eos_token_id:
                break  # 如果生成EOS token，提前结束
            
            generated_content += [generated_token]
    torch.cuda.synchronize()
    end_decode = time.time()
    decode_time = end_decode - start_decode
    print(f"Prefill Time: {prefill_time:.4f} s")
    print(f"Decode Time: {decode_time:.4f} s")
    output_text = tokenizer.decode(generated_content, skip_special_tokens=True)
    print("Generated Text:", output_text)
