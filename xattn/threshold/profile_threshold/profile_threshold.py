from transformers import LlamaForCausalLM,AutoTokenizer
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import Cache,LlamaForCausalLM,repeat_kv,apply_rotary_pos_emb,logger
from tqdm import tqdm
from flash_attn import flash_attn_func
from xattn.src.utils import create_causal_mask
import json
from xattn.src.Xattention import xattn_estimate
import torch.nn.functional as F
import math, argparse


def chunk_prefill_to_attn_sum(query_states:torch.Tensor,key_states:torch.Tensor,block_size)->torch.Tensor:
    batch_size,num_kv_head,k_len,head_dim = key_states.shape
    batch_size,num_q_head,q_len,head_dim = query_states.shape
    q_num_to_pad = ((q_len+block_size-1)//block_size)*block_size - q_len
    k_num_to_pad = ((k_len+block_size-1)//block_size)*block_size - k_len
    q_block_num = (q_len+block_size-1)//block_size
    k_block_num = (k_len+block_size-1)//block_size
    key_states = F.pad(key_states,(0,0,0,k_num_to_pad),value = 0).to("cuda")
    query_states = F.pad(query_states,(0,0,0,q_num_to_pad),value = 0).to("cuda")

    decoding = (q_block_num!=k_block_num)
    assert(not decoding)
    assert(num_kv_head == num_q_head)
    attn_sum_list = []
    for i in range(q_block_num):
        query_states_slice = query_states[:,:,i*block_size:(i+1)*block_size,:].to("cuda")
        attn_weights_slice = torch.matmul(query_states_slice,key_states.transpose(2,3)/ math.sqrt(head_dim)).to("cuda")
        
        causal_mask = create_causal_mask(batch_size,num_kv_head,block_size,k_block_num,i)
        causal_mask = F.pad(causal_mask[:,:,:,0:k_len],(0,k_num_to_pad),value=float("-inf"))
        
        if(i==q_block_num-1):
            causal_mask = F.pad(causal_mask[:,:,0:block_size-q_num_to_pad,:],(0,0,0,q_num_to_pad),value=float("-inf"))
                    
        attn_weights_slice = attn_weights_slice + causal_mask.to(attn_weights_slice.device)

        attn_weights_slice = F.softmax(attn_weights_slice, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

        if i == q_block_num - 1:
            attn_weights_slice = F.pad(attn_weights_slice[:,:,0:block_size-q_num_to_pad,:],(0,0,0,q_num_to_pad))

        attn_sum = attn_weights_slice.view(batch_size,num_kv_head,block_size,-1,block_size).sum(dim=-1).sum(dim=-2).to("cuda")
        
        attn_sum_list.append(attn_sum.unsqueeze(dim=-2))

        del attn_weights_slice

    attn_sums = torch.cat(attn_sum_list,dim = -2)

    return attn_sums

def x_attn_map(query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    chunk_size=16384,
    causal=True):
    return

def xattn_prefill_profile(self, query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states,
    block_size,
    stride,
    chunk_size=16384,
    causal=True
    ):
        attn_sum = chunk_prefill_to_attn_sum(query_states=query_states,key_states=key_states,block_size=block_size)

        xattn_sum,_ = xattn_estimate(query_states,
                key_states,
                block_size=block_size,
                stride=stride,
                norm=1,
                threshold=1,
                select_mode="inverse",
                use_triton=True,
                causal=causal,
                chunk_size=chunk_size
            )
        xattn_sum = xattn_sum[:,:,:attn_sum.shape[-1],:attn_sum.shape[-1]] * stride

        # ---------- step-1: pick the fewest blocks that cover ≥ 90 % of attn_sum ----------
        vals, idx = torch.sort(attn_sum, dim=-1, descending=True)      # sort each row in descending order
        cumsum    = vals.cumsum(-1)
        need_len  = (cumsum < 0.9 * cumsum[..., -1:]).sum(-1) + 1      # number of blocks needed per row
        rank_idx  = torch.arange(vals.size(-1), device=vals.device)
        sel_mask_srt = rank_idx.view(1, 1, 1, -1) < need_len.unsqueeze(-1)  # mask of the selected (sorted) blocks

        # ---------- step-2: locate miniblocks in xattn_sum and compute row-wise thresholds ----------
        x_sorted    = xattn_sum.gather(-1, idx)                         # reorder xattn_sum in the same way as vals
        miniblock   = x_sorted.masked_fill(~sel_mask_srt, float("inf")).min(-1).values
        mask_ge_min = xattn_sum >= miniblock.unsqueeze(-1)              # blocks whose value ≥ miniblock
        threshold_head = (xattn_sum * mask_ge_min).sum(-1).sum(-1) / (xattn_sum.sum(-1).sum(-1) + 1e-8)


        if self.layer_idx == 0:
            self.profile_config.history_threshold.append(threshold_head)
        else:
            self.profile_config.history_threshold[-1] = torch.concat([self.profile_config.history_threshold[-1],threshold_head],dim=0)

        return flash_attn_func(q = query_states.transpose(1,2),k = key_states.transpose(1,2),v = value_states.transpose(1,2), causal = causal).transpose(1,2).contiguous()

def forward_profile(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
   
        _, _, k_len, _ = key_states.shape
        _, _, q_len, _ = query_states.shape
        decoding = (q_len != k_len and q_len == 1)
        if not decoding:
            key_states = repeat_kv(key_states, self.num_key_value_groups).to("cuda")
            value_states = repeat_kv(value_states, self.num_key_value_groups).to("cuda")
        
        stride = self.profile_config.stride
        causal = self.profile_config.causal
        # Profiling Xattention
        # Prefilling only
        assert (key_states.shape == query_states.shape)
        attn_output = xattn_prefill_profile(self,query_states = query_states,key_states=key_states,value_states=value_states,block_size=128,stride=stride,causal = causal)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        del query_states
        
        return attn_output, None, past_key_value

class ProfileConfig():
    def __init__(self, stride = 8, causal=True):
        self.stride = stride
        self.history_threshold = []
        self.causal = causal
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--causal", type=bool, default=True)
    args = parser.parse_args()

    name_or_path = args.name_or_path

    model = LlamaForCausalLM.from_pretrained(
        name_or_path,
        device_map="balanced", 
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    profile_config = ProfileConfig(stride=args.stride,causal = args.causal)

    for layer in model.model.layers:
        layer.self_attn.forward = forward_profile.__get__(layer.self_attn)
        layer.self_attn.profile_config = profile_config
    
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path
    )

    with open("xattn/threshold/profile_threshold/text.json", "r") as f:
        texts = json.load(f)
    num = 0
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            _ = model(**inputs)
        num += 1

    history_threshold = model.model.layers[-1].self_attn.profile_config.history_threshold
    final_threshold = torch.concat([threshold.unsqueeze(0) for threshold in history_threshold]).max(0)[0]
    print(final_threshold.tolist)