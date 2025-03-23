from xattn.src.utils import *
import torch
import math
import torch.nn.functional as F
from xattn.src.kernels import (
    flat_group_gemm,
    softmax_fuse_block_sum,
    flat_group_gemm_fuse_reshape,
)
from block_sparse_attn import block_sparse_attn_func


def xattn_estimate(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    block_size,
    stride,
    norm=1,
    softmax=True,
    threshold=0.9,
    chunk_size=16384,
    select_mode="inverse",
    use_triton=True,
    causal=True,
    kdb: int = 1,
    keep_sink=False,
    keep_recent=False,
) -> torch.Tensor:
    batch_size, num_kv_head, k_len, head_dim = key_states.shape
    batch_size, num_q_head, q_len, head_dim = query_states.shape
    assert num_q_head == num_kv_head

    k_num_to_pad = ((k_len + chunk_size - 1) // chunk_size) * chunk_size - k_len
    q_num_to_pad = ((q_len + chunk_size - 1) // chunk_size) * chunk_size - q_len
    k_chunk_num = (k_len + k_num_to_pad) // chunk_size
    k_block_num = (k_len + k_num_to_pad) // block_size
    q_chunk_num = (q_len + q_num_to_pad) // chunk_size
    q_block_num = (q_len + q_num_to_pad) // block_size

    if k_num_to_pad > 0:
        pad_key_states = F.pad(key_states, (0, 0, 0, k_num_to_pad), value=0).to("cuda")
    else:
        pad_key_states = key_states
    if q_num_to_pad > 0:
        pad_query_states = F.pad(query_states, (0, 0, 0, q_num_to_pad), value=0).to(
            "cuda"
        )
    else:
        pad_query_states = query_states

    assert num_kv_head == num_q_head
    attn_sum_list = []
    simple_mask_list = []

    if use_triton and (
        "100" not in torch.cuda.get_device_properties(torch.cuda.current_device()).name
    ):
        use_triton = False
        print(
            "setting use triton to false. Triton kernel not surpported on this device"
        )

    reshaped_chunk_size = chunk_size // stride
    reshaped_block_size = block_size // stride
    k_reshaped_num_to_pad = k_num_to_pad // stride
    k_reshaped_seq_len = (k_len + k_num_to_pad) // stride
    q_reshaped_num_to_pad = q_num_to_pad // stride
    num_blocks_per_chunk = reshaped_chunk_size // reshaped_block_size
    if not use_triton:
        if select_mode == "random":
            perm_idx = torch.randperm(stride)
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    pad_query_states[:, :, perm_idx[i] :: stride, :]
                    for i in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "inverse" or select_mode == "":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: (stride * kdb), :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "slash":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_query = torch.cat(
                [(pad_query_states[:, :, q::stride, :]) for q in range(stride)], dim=-1
            )
        elif select_mode == "double":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        elif select_mode == "triple":
            reshaped_key = torch.cat(
                [(pad_key_states[:, :, k::stride, :]) for k in range(stride)], dim=-1
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, head_dim:], reshaped_key[:, :, :, 0:head_dim]],
                dim=-1,
            )
            reshaped_key = reshaped_key + torch.cat(
                [reshaped_key[:, :, :, -head_dim:], reshaped_key[:, :, :, 0:-head_dim]],
                dim=-1,
            )
            reshaped_query = torch.cat(
                [
                    (pad_query_states[:, :, (stride - 1 - q) :: stride, :])
                    for q in range(stride)
                ],
                dim=-1,
            )
        assert reshaped_key.shape[-2] == k_reshaped_seq_len

    for chunk_idx in range(q_chunk_num):
        if use_triton:
            if kdb != 1:
                raise ValueError("use_triton and kdb cannot be used together")
            attn_weights_slice = flat_group_gemm_fuse_reshape(
                pad_query_states[
                    :,
                    :,
                    (chunk_idx * reshaped_chunk_size)
                    * stride : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                    * stride,
                    :,
                ],
                pad_key_states,
                stride,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                is_causal=causal,
            )
            attn_sum = softmax_fuse_block_sum(
                attn_weights_slice,
                reshaped_block_size,
                min(4096, reshaped_block_size),
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size,
                (k_block_num - q_block_num) * reshaped_block_size
                + chunk_idx * reshaped_chunk_size
                + reshaped_chunk_size,
                k_reshaped_seq_len - k_reshaped_num_to_pad,
                1.4426950408889634 / math.sqrt(head_dim) / stride / norm,
                is_causal=causal,
            )
        else:
            chunked_query = reshaped_query[
                :,
                :,
                (chunk_idx * reshaped_chunk_size)
                // kdb : (chunk_idx * reshaped_chunk_size + reshaped_chunk_size)
                // kdb,
                :,
            ]
            attn_weights_slice = torch.matmul(
                chunked_query,
                reshaped_key.transpose(2, 3),
            ).to("cuda")

            attn_weights_slice = (
                attn_weights_slice / math.sqrt(head_dim) / stride / norm
            )

            if causal:
                causal_mask = torch.zeros(
                    (
                        batch_size,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size * k_chunk_num,
                    ),
                    device=key_states.device,
                )
                causal_mask[:, :, :, (-k_reshaped_num_to_pad):] = float("-inf")
                chunk_start = chunk_idx * reshaped_chunk_size
                chunk_end = chunk_start + reshaped_chunk_size
                causal_mask[:, :, :, chunk_start:chunk_end] = torch.triu(
                    torch.ones(
                        1,
                        num_q_head,
                        reshaped_chunk_size,
                        reshaped_chunk_size,
                        device=key_states.device,
                    )
                    * float("-inf"),
                    diagonal=1,
                )

                if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                    causal_mask[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = float(
                        "-inf"
                    )

                causal_mask[:, :, :, chunk_end:] = float("-inf")
                causal_mask = causal_mask[:, :, kdb - 1 :: kdb, :]
                attn_weights_slice = attn_weights_slice + causal_mask.to(
                    attn_weights_slice.device
                )

            if softmax:
                attn_weights_slice = F.softmax(
                    attn_weights_slice, dim=-1, dtype=torch.float32
                ).to(pad_query_states.dtype)
            else:
                attn_weights_slice = torch.exp(attn_weights_slice).to(
                    pad_query_states.dtype
                )
            attn_weights_slice = F.dropout(attn_weights_slice, p=0, training=False)

            if chunk_idx == q_chunk_num - 1 and q_reshaped_num_to_pad != 0:
                attn_weights_slice[:, :, (-(q_reshaped_num_to_pad // kdb)) :, :] = 0

            attn_sum = (
                attn_weights_slice.view(
                    batch_size,
                    num_kv_head,
                    num_blocks_per_chunk,
                    reshaped_block_size // kdb,
                    -1,
                    reshaped_block_size,
                )
                .sum(dim=-1)
                .sum(dim=-2)
                .to("cuda")
            )
            del chunked_query
        
        simple_mask = find_blocks_chunked(
            attn_sum,
            k_block_num - q_block_num + chunk_idx * num_blocks_per_chunk,
            threshold,
            None,
            decoding=False,
            mode="prefill",
            causal=causal,
        )

        attn_sum_list.append(attn_sum)
        simple_mask_list.append(simple_mask)

        del attn_weights_slice

    if not use_triton:
        del reshaped_query, reshaped_key
    attn_sums = torch.cat(attn_sum_list, dim=-2)
    simple_masks = torch.cat(simple_mask_list, dim=-2)

    if causal:
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            torch.tril(
                torch.ones(
                    q_block_num, q_block_num, dtype=bool, device=key_states.device
                ),
                diagonal=0,
            ),
            simple_masks[:, :, -q_block_num:, -q_block_num:],
            False,
        )
    if keep_sink:
        simple_masks[:, :, 0, :] = True
    if keep_recent:
        eye_matrix = torch.eye(q_block_num, device=simple_masks.device, dtype=bool)
        eye_matrix_expanded = (
            eye_matrix.unsqueeze(0)
            .unsqueeze(0)
            .expand(1, num_kv_head, q_block_num, q_block_num)
        )
        simple_masks[:, :, -q_block_num:, -q_block_num:] = torch.where(
            eye_matrix_expanded, True, simple_masks[:, :, -q_block_num:, -q_block_num:]
        )

    return attn_sums, simple_masks


def Xattention_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    stride,
    norm=1,
    threshold=0.8,
    block_size=128,
    use_triton=True,
    causal=True,
    kdb=1,
    chunk_size=None,
    keep_sink=False,
    keep_recent=False,
):
    batch_size, num_heads, k_len, head_dim = key_states.shape
    _, _, q_len, _ = query_states.shape

    q_block_num = (q_len + block_size - 1) // block_size
    k_block_num = (k_len + block_size - 1) // block_size
    if chunk_size is None:
        chunk_size = int(
            max(
                min(
                    max(2048, 1 << (k_len - 1).bit_length()),
                    128 * 1024 * 2048 // (1 << (k_len - 1).bit_length()),
                ),
                2048,
            )
        )
    attn_sums, approx_simple_mask = xattn_estimate(
        query_states,
        key_states,
        block_size=block_size,
        stride=stride,
        norm=norm,
        threshold=threshold,
        select_mode="inverse",
        use_triton=use_triton,
        causal=causal,
        chunk_size=chunk_size,
        kdb=kdb,
        keep_sink=keep_sink,
        keep_recent=keep_recent,
    )

    if query_states.device != key_states.device:
        key_states = key_states.to(query_states.device)
    if query_states.device != value_states.device:
        value_states = value_states.to(query_states.device)
    if approx_simple_mask.device != query_states.device:
        approx_simple_mask = approx_simple_mask.to(query_states.device)

    ####################
    assert block_size == 128
    assert batch_size == 1
    query_states = query_states.transpose(1, 2).view(q_len, num_heads, head_dim)
    key_states = key_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    value_states = value_states.transpose(1, 2).view(k_len, num_heads, head_dim)
    q_cu_seq_lens = torch.tensor(
        [0, q_len], dtype=torch.int32, device=query_states.device
    )
    k_cu_seq_lens = torch.tensor(
        [0, k_len], dtype=torch.int32, device=query_states.device
    )
    head_mask_type = torch.tensor(
        [1 for _ in range(num_heads)], device=query_states.device, dtype=torch.int32
    )
    assert head_mask_type.device == query_states.device
    assert q_cu_seq_lens.device == query_states.device
    assert k_cu_seq_lens.device == query_states.device
    assert key_states.device == query_states.device
    assert value_states.device == query_states.device
    assert approx_simple_mask.device == query_states.device

    attn_output = block_sparse_attn_func(
        query_states,
        key_states,
        value_states,
        q_cu_seq_lens,
        k_cu_seq_lens,
        head_mask_type,
        None,
        approx_simple_mask[:, :, :q_block_num, :k_block_num].contiguous(),
        q_len,
        k_len,
        p_dropout=0.0,
        deterministic=True,
        is_causal=causal,
    )
    attn_output = attn_output.view(batch_size, q_len, num_heads, head_dim).transpose(
        1, 2
    )
    ################################

    del query_states
    num_to_compute = (k_block_num + 1) * k_block_num / 2 * num_heads
    
    # print(f"approximated prefilling Computation: {approx_simple_mask.sum() / num_to_compute}")
    del approx_simple_mask, attn_sums
    return attn_output
