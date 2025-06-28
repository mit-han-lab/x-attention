try:
    from xattn.src.Xattention import Xattention_prefill
    XATTN_PREFILL = True
except:
    XATTN_PREFILL = False

try:
    from xattn.src.Flexprefill import Flexprefill_prefill
    FLEXPREFILL_PREFILL = True
except:
    FLEXPREFILL_PREFILL = False

try:
    from xattn.src.Minference import Minference_prefill
    MINFERENCE_PREFILL = True
except:
    MINFERENCE_PREFILL = False

try:
    from xattn.src.Fullprefill import Full_prefill
    FULL_PREFILL = True
except:
    FULL_PREFILL = False
import pickle
import torch
import time
from eval.efficiency.generate_prompt import generate_prompt
from xattn.src.load_llama import load_fake_model,FastPrefillConfig
from xattn.threshold.llama_threshold import llama_fuse_8,llama_fuse_16
from transformers import StaticCache
from tqdm import tqdm
import os

if __name__ == "__main__":

    lens = [4,8,16,32,64,128]

    speedups_flex = []
    speedups_xattn_8 = []
    speedups_xattn_16 = []
    speedups_minfer = []
    past_key_values = None
    for len in lens:
        print(f"Testing {len}K")
        query_path = f"output/query_{len*1024}.pkl"
        key_path = f"output/key_{len*1024}.pkl"
        config = FastPrefillConfig(metric = "xattn",stride = 16)
        layer_to_save = 12
        if not os.path.exists(query_path) or not os.path.exists(key_path):
            
            model, tokenizer = load_fake_model(name_or_path="meta-llama/Llama-3.1-8B-Instruct", layer_to_save=layer_to_save, target_len=len*1024)
            input_ids = generate_prompt(tokenizer,len*1024)
            chunk_size = 4096
            if past_key_values is not None:
                past_key_values.reset()
            else:
                past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=300000, device=model.device, dtype=model.dtype)
            with torch.no_grad():
                for i in tqdm(range(0, input_ids.size(1), chunk_size), desc="Prefilling", unit="chunk"):
                    chunk = input_ids[:, i: i + chunk_size]
                    output = model(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                        num_logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
        with open(query_path, "rb") as f:
            q = pickle.load(f)
        with open(key_path, "rb") as f:
            k = pickle.load(f)
        assert(q.shape[-2] == len*1024)
        assert(k.shape[-2] == len*1024)
        torch.manual_seed(0)
        # FlexPrefill args
        gamma = 0.95
        tau = 0.1
        # Xattention args
        threshold = torch.tensor(llama_fuse_8)[layer_to_save]
        stride = 16
        v = torch.randn(q.shape, dtype=torch.bfloat16).to("cuda").contiguous()
        num_iterations = 50
        num_warmups = 30
        # warm up
        for i in range(num_warmups):
            try:
                Xattention_prefill(q, k, v, stride=16, threshold=threshold, use_triton=True)
                Xattention_prefill(q, k, v, stride=8, threshold=threshold, use_triton=True)
            except:
                XATTN_PREFILL = False
            try:
                Full_prefill(q, k, v)
            except:
                FULL_PREFILL = False
            try:
                Flexprefill_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gamma, tau)
            except:
                FLEXPREFILL_PREFILL = False
            try:
                Minference_prefill(k, q, v)
            except:
                MINFERENCE_PREFILL = False

        # Efficiency Evaluation
        # For Flexprefill_prefill
        total_time_flex = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            flex_prefill_output = Flexprefill_prefill(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), gamma, tau)
            torch.cuda.synchronize()
            total_time_flex += time.time() - start_time
        avg_time_flex = total_time_flex / num_iterations

        # For Xattention_prefill
        total_time_xattn_8 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            flex_prefill_output = Xattention_prefill(q, k, v, stride=8, threshold= threshold, use_triton=True,chunk_size=min(32768,len*1024))
            torch.cuda.synchronize()
            total_time_xattn_8 += time.time() - start_time
        avg_time_xattn_8 = total_time_xattn_8 / num_iterations

        total_time_xattn_16 = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            flex_prefill_output = Xattention_prefill(q, k, v, stride=16, threshold= threshold, use_triton=True,chunk_size=min(32768,len*1024))
            torch.cuda.synchronize()
            total_time_xattn_16 += time.time() - start_time
        avg_time_xattn_16 = total_time_xattn_16 / num_iterations

        # For minference
        total_time_minfer = 0
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            if MINFERENCE_PREFILL:
                try:
                    flex_prefill_output = Minference_prefill(k, q, v)
                except:
                    MINFERENCE_PREFILL = False
            torch.cuda.synchronize()
            total_time_minfer += time.time() - start_time
        avg_time_minfer = total_time_minfer / num_iterations

        # For flashinfer
        total_time_flashinfer = 0
        torch.cuda.synchronize()
        start_time = time.time()
        o = Full_prefill(q, k, v, causal=False)
        torch.cuda.synchronize()
        total_time_flashinfer += time.time() - start_time
        avg_time_flashinfer = total_time_flashinfer

        # Calculate speedups
        print(f"{len}K Minfer {avg_time_minfer:.4f} flex: {avg_time_flex:.4f} xattn_8: {avg_time_xattn_8:.4f} xattn_16: {avg_time_xattn_16:.4f} full: {avg_time_flashinfer:.4f} ")
        speedup_flex = avg_time_flashinfer / avg_time_flex
        speedup_xattn_8 = avg_time_flashinfer / avg_time_xattn_8
        speedup_xattn_16 = avg_time_flashinfer / avg_time_xattn_16
        speedup_minfer = avg_time_flashinfer / avg_time_minfer
        speedups_flex.append(speedup_flex)
        speedups_xattn_8.append(speedup_xattn_8)
        speedups_xattn_16.append(speedup_xattn_16)
        speedups_minfer.append(speedup_minfer)

    # Output results
    print(f"\n{'Length':<10}{'Flex Speedup':<15}{'Xattn 8 Speedup':<20}{'Xattn 16 Speedup':<25}{'Minfer Speedup'}")
    for len, speedup_flex, speedup_xattn_8, speedup_xattn_16,speedup_minfer in zip(lens, speedups_flex, speedups_xattn_8, speedups_xattn_16, speedups_minfer):
        print(f"{str(len):<10}{speedup_flex:<15.2f}{speedup_xattn_8:<20.2f}{speedup_xattn_16:<25.2f}{speedup_minfer:.2f}")