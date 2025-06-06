import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from utils import torch_randint


G_device = "cuda" if torch.cuda.is_available() else "cpu"
G = torch.tensor([
    [2, -1,   0,   0,   0,   0,   0,  0.5],
    [0,  1,  -1,   0,   0,   0,   0,  0.5],
    [0,  0,   1,  -1,   0,   0,   0,  0.5],
    [0,  0,   0,   1,  -1,   0,   0,  0.5],
    [0,  0,   0,   0,   1,  -1,   0,  0.5],
    [0,  0,   0,   0,   0,   1,  -1,  0.5],
    [0,  0,   0,   0,   0,   0,   1,  0.5],
    [0,  0,   0,   0,   0,   0,   0,  0.5]
], dtype=torch.float32, device=G_device)


G_inv = torch.inverse(G)
EPS = 5e-7
EPS_vector = torch.tensor([i * EPS for i in range(8)], device=G_device, dtype=torch.float32)


def encode_d8(x):
    x_floor = torch.floor(x + EPS)
    d_floor = (x - x_floor)**2
    d_floor1 = (x - (x_floor + 1))**2
    mask = (d_floor < d_floor1 + EPS)
    eps_sign = torch.where(mask, -1, 1)
    opt = torch.where(mask, x_floor, x_floor + 1)

    sum_opt = torch.round(torch.sum(opt, dim=1))
    need_to_fix = (sum_opt % 2 == 1)
    fix_cost = torch.abs(d_floor - d_floor1)

    if need_to_fix.any():
        fix_cost_rows = fix_cost[need_to_fix]  # (num_fix,8)
        fix_loc = torch.argmin(fix_cost_rows + eps_sign[need_to_fix] * EPS_vector[None, :], dim=1)
        idx_rows = torch.nonzero(need_to_fix).flatten()
        opt[idx_rows, fix_loc] = (2 * x_floor[idx_rows, fix_loc] + 1) - opt[idx_rows, fix_loc]

    return opt
    

def row_norm(x1, x2):
    return torch.sum((x1 - x2)**2, dim=1)


def encode_e8(x):
    return encode_e8_fast(x)
    x1 = encode_d8(x)
    x2 = encode_d8(x - 0.5) + 0.5
    r1 = row_norm(x1, x)
    r2 = row_norm(x2, x)
    first_win = r1 < r2 - EPS
    eq = ~first_win & (r1 < r2 + EPS)
    return torch.where((first_win | (eq & (x1[:, 0] < x2[:, 0])))[:, None], x1, x2)


def generate_dither(n, generator, device):
    U = torch.rand((n, 8), generator=generator, device=device, dtype=torch.float32) * 2.0
    return U - encode_e8(U)


def encode_e8_fast(x):
    N = x.shape[0]
    d = torch.floor(x)
    g = x > (d + 0.5)
    opt = d + g
    opt2 = d + 0.5
    bad = torch.sum(opt, dim=1).to(torch.int32) & 1
    bad2 = torch.sum(opt2, dim=1).to(torch.int32) & 1
    dist_to_half = (opt2 - x) * (1 - 2 * g)

    flip1 = torch.zeros_like(x)
    flip1[torch.arange(N), torch.argmin(dist_to_half, dim=1)] = 1.0
    flip1 = flip1 * bad[:, None]

    flip2 = torch.zeros_like(x)
    flip2[torch.arange(N), torch.argmax(dist_to_half, dim=1)] = 1.0
    flip2 = flip2 * bad2[:, None]

    opt = flip1 * (2 * d + 1 - opt) + (1 - flip1) * opt
    opt2 = flip2 * (opt2 - 1 + 2 * g) + (1 - flip2) * opt2

    mse = ((opt - x) ** 2).sum(dim=1)
    mse2 = ((opt2 - x) ** 2).sum(dim=1)
    return torch.where((mse < mse2)[:, None], opt, opt2)
