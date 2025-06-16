import torch
from torch import nn
import torch.distributed as dist
import numpy as np
from torch.distributed import get_world_size, get_rank
from fairscale.nn.model_parallel.layers import RowParallelLinear

from coder import encode_matrix, decode_matrix
from hadamard import kron_h_ip


def rot_id(X, inverse=False):
    return X


def rot_hess(H, rot, eye_coeff=100.0):
    H = H.T
    H = rot(H)
    H = H.T
    H = rot(H)
    H = H + torch.eye(H.shape[0], device=H.device) * eye_coeff
    return H

def quantsim(X, q, betas, rot, H=None, eps=None, J=None):
    assert J is None or eps is None, "Cannot use eps and J at the same time"
    orig_shape = X.shape
    X = X.view(-1, X.shape[-1])
    original_dtype = X.dtype
    X = X.float()  # Convert to float32 for numerical ops

    if J is not None:
        J = J.float()
        I = torch.eye(H.shape[0], device=X.device, dtype=H.dtype)
        # print(f"{J.sum()=}, {J.diag().sum()=}")
        # eps2 = max((J.sum().item() / J.shape[0]), (H**2).mean().item() * 1e-6)
        # X = X @ H @ torch.linalg.inv(H + eps2 * I)
        print(J.diag()[:100].tolist())
        X = X @ H @ torch.linalg.inv(H + J.diag().diag())
        H = H.float() + J.diag().diag()
 
    elif eps is not None:
        H = H.float()
        eps2 = eps * eps
        n = X.shape[-1]
        I = torch.eye(n, device=X.device, dtype=H.dtype)
        X = X @ (I - eps2 * torch.linalg.inv(H + eps2 * I))
        H = H + I * eps2

    row_norms = torch.sqrt((X ** 2).sum(dim=1))
    N = X.shape[1]
    X = X / row_norms[:, None] * np.sqrt(N)
    X = rot(X)
    if H is not None:
        H = rot_hess(H, rot)

    with torch.inference_mode():
        use_ldlq = H is not None
        X_enc = encode_matrix(X, q, betas, 0, use_dither=False, try_all=True, H=H)
        X = decode_matrix(X_enc, q, betas, 0, use_dither=False, try_all=True, use_ldlq=use_ldlq)

    X = rot(X, inverse=True)
    X = X * row_norms[:, None] / np.sqrt(N)
    X = X.reshape(orig_shape)
    return X.to(original_dtype)  # Restore original dtype (e.g., bfloat16)

# The quantization of rows can be done independetly on each GPU.
# However, sometimes the matrix is split by columns. So, we gather the matrix,
# split by rows, each GPU quantizes its rows, and we re-split it by columns
def re_glue(X, predicate):
    world_size = get_world_size()
    rank = get_rank()
    tensor_list = [torch.empty_like(X) for _ in range(world_size)]
    dist.all_gather(tensor_list, X)
    all_X = torch.cat(tensor_list, dim=-1)

    assert all_X.shape[-2] % world_size == 0
    chunk = torch.chunk(all_X, world_size, dim=-2)[rank]
    chunk = predicate(chunk)

    tensor_list = [torch.empty_like(chunk) for _ in range(world_size)]
    dist.all_gather(tensor_list, chunk)
    all_X = torch.cat(tensor_list, dim=-2)
    return torch.chunk(all_X, world_size, dim=-1)[rank]


def quantsim_col(X, q, betas, rot, H=None, eps=None, J=None):
    return re_glue(X, lambda x: quantsim(x, q, betas, rot, H=H, eps=eps, J=J))


def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return float('nan') if ss_tot == 0 else (1 - ss_res / ss_tot).item()


def do_quant_act_hook(_, input, is_col, act_config):
    X = input[0].float()
    if is_col:
        X = quantsim_col(X, act_config.q, act_config.betas, kron_h_ip)
    else:
        X = quantsim(X, act_config.q, act_config.betas, kron_h_ip)
    X_res = X.to(torch.bfloat16)
    return (X_res,)


def get_quant_act_hook(module, act_config):
    is_col = isinstance(module, RowParallelLinear)
    return lambda module, input: do_quant_act_hook(module, input, is_col, act_config)
