import torch
import torch.nn.functional as F
import math
from minference.ops.pit_sparse_flash_attention_v2 import (
    vertical_slash_sparse_attention,
)



last_q = 64
arange = torch.arange(last_q, device="cuda")
LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    zero_mat = torch.zeros((b, h, n, n)).to(mat.device)  # Zero matrix used for padding
    mat_padded = torch.cat(
        (zero_mat, mat, zero_mat), -1
    )  # pads the matrix on left and right
    mat_strided = mat_padded.as_strided(
        (1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1)
    )  # Change the strides
    sum_diags = torch.sum(mat_strided, 2)  # Sums the resulting matrix's columns
    return sum_diags[:, :, 1:]



def Minference_prefill(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    vertical_size=1000,
    slash_size=6096,
    adaptive_budget=None,
):
    output = torch.empty_like(query_states)
    key_states.to(query_states.device)
    value_states.to(query_states.device)
    if adaptive_budget is not None:
        seq_len = query_states.size(2)
        budget = int(seq_len * adaptive_budget)
        vertical_size = int(budget * 0.2)
        slash_size = int(budget * 0.8)
    for head in range(query_states.size(1)):
        q = query_states[:, head, :, :].unsqueeze(1).to(query_states.device)
        k = key_states[:, head, :, :].unsqueeze(1).to(query_states.device)
        v = value_states[:, head, :, :].unsqueeze(1).to(query_states.device)

        q_len = q.shape[2]
        vertical_size, slash_size = min(q_len, max(vertical_size, 30)), min(
            q_len, max(slash_size, 50)
        )
        last_q = min(64, q_len)
        qk = torch.einsum(f"bhmk, bhnk -> bhmn", q[:, :, -last_q:, :], k) / math.sqrt(
            128
        )  # headdim
        qk[:, :, :, -last_q:] = torch.where(
            LAST_Q_MASK[..., -last_q:, -last_q:].to(q.device),
            qk[:, :, :, -last_q:],
            -torch.inf,
        )
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[..., : -last_q + 1]
        slash[..., -100:] = torch.inf
        slash_topk = slash
        slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        output[:, head : head + 1, :, :] = vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)

    return output
