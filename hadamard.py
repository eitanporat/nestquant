import torch
import numpy as np
from einops import rearrange

from utils import hadamard_transform_adaptive, multiply_kronecker, is_pow_of_2
from h_matrices import get_had28, get_had172, get_had20, get_had108


device = "cuda" if torch.cuda.is_available() else "cpu"

mat = {
    20: get_had20().to(device),
    28: get_had28().to(device),
    108: get_had108().to(device),
    172: get_had172().to(device),
}


def had_kron(x, K, inverse=False):
    assert x.shape[-1] % K == 0
    assert is_pow_of_2(x.shape[-1] // K)
    n1 = x.shape[-1] // K
    had1 = lambda x: hadamard_transform_adaptive(x, scale= 1.0 / np.sqrt(n1))
    H2 = mat[K].T if inverse else mat[K]
    had2 = lambda x: x @ H2.to(x.device) / np.sqrt(K)
    return multiply_kronecker(x, K, had2, n1, had1)


def rotate_fft(x):
    x = rearrange(x, "b (d c) -> b d c", c=2)
    x = torch.view_as_complex(x)
    x = torch.fft.fft(x, dim=1) / np.sqrt(x.shape[-1])
    x = torch.view_as_real(x)
    x = rearrange(x, "b d c -> b (d c)")
    return x


def kron_h_ip(x, inverse=False):
    N = x.shape[1]
    if N == 14336 or N == 28672:
        return had_kron(x, 28, inverse=inverse)
    elif N == 11008:
        return had_kron(x, 172, inverse=inverse)
    elif N == 5120:
        return had_kron(x, 20, inverse=inverse)
    elif N == 13824:
        return had_kron(x, 108, inverse=inverse)
    elif is_pow_of_2(N):
        return hadamard_transform_adaptive(x, 1.0 / np.sqrt(N))
    else:
        raise ValueError("Bad N")


def kron_H_hess(H):
    H = H.T
    H = kron_h_ip(H)
    H = H.T
    H = kron_h_ip(H)
    H = H + torch.eye(H.shape[0], device=H.device) * 100.0
    return H
