from einops import rearrange
from collections import defaultdict
import torch
import re
import matplotlib.pyplot as plt
import math
import gc
import itertools
import os
import torch.nn.functional as F
import scipy

if torch.cuda.is_available():
    from fast_hadamard_transform import hadamard_transform


def plot_error(approx, real, title="Relative error distribution", ax=None):
    relative_error = ((approx - real).abs() / ((real).abs() + 1e-5)).flatten()
    relative_error = relative_error[torch.abs(relative_error) > 1e-5]
    log_relative_error = torch.log10(relative_error + 1e-10)
    if ax is None:
        fig, ax = plt.subplots(1,1)
        
    ax.set_xlabel("Log of relative error")
    ax.set_title(title)
    ax.hist(log_relative_error.detach().cpu().numpy(), bins=100)
    plt.show()


def count_parameters_by_type(model):
    param_count_by_type = defaultdict(int)

    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_type = re.sub(r'(\d+)', 'X', name)
            param_count_by_type[layer_type] += param.numel()

    for layer_type, total_count in sorted(list(param_count_by_type.items()), key=lambda x: x[1], reverse=True):
        print(f"{layer_type}: {total_count / 1_000_000:.2f}M")


def matrix_to_pieces(A):
    # Splits rows of A into pieces of 8
    return rearrange(A, "r (c b) -> (r c) b", b=8)


def pieces_to_matrix(A, shape):
    # Restores A from pieces
    return rearrange(A, "(r c) b -> r (c b)", r=shape[0])


def torch_randint(l, r):
    return torch.randint(l, r, (1,)).item()


def generate_seed():
    return torch_randint(0, 2 ** 32)


def generate_orthonormal(n, device):
    random_matrix = torch.randn((n, n), device=device)
    q, r = torch.linalg.qr(random_matrix)
    d = torch.diag(torch.sign(torch.diag(r)))
    orthonormal_matrix = q @ d
    return orthonormal_matrix


def is_pow_of_2(n):
    return (n & (n - 1)) == 0


def next_pow_of_2(n):
    res = 1
    while res < n:
        res <<= 1
    return res


def get_size_for_dtype(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def get_tensor_counts():
    tensor_info = {}
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda:
            key = (obj.size(), obj.dtype)
            if key not in tensor_info:
                tensor_info[key] = 0
            tensor_info[key] += 1
    return tensor_info


def get_size(kv_pair):
    key, cnt = kv_pair
    shape, dtype = key
    return math.prod(shape) * get_size_for_dtype(dtype) * cnt / (1024 ** 2)


def print_tensor_counts(tensor_counts):
    print(f"{'Shape':<30} {'Memory (MB)':<15} {'DType':<20} {'Count':<15}")
    for key, cnt in sorted(tensor_counts.items(), key=get_size, reverse=True):
        shape, dtype = key
        print(f"{str(shape):<30} {get_size((key, cnt)):<15.2f} {str(dtype):<20} {cnt:<15}")


def non_zero_differences(dict1, dict2):
    all_keys = set(dict1.keys()).union(dict2.keys())

    result = {
        key: dict1.get(key, 0) - dict2.get(key, 0)
        for key in all_keys
        if dict1.get(key, 0) - dict2.get(key, 0) != 0
    }
    return result


def int_tensor_to_counts(x, target_size=None):
    max_value = x.max().item()
    min_value = x.min().item()
    assert min_value >= 0
    if target_size is None:
        target_size = max_value + 1
    else:
        assert target_size > max_value
    res = torch.zeros((target_size,), dtype=torch.long, device=x.device)
    for i in range(target_size):
        res[i] = (x == i).sum().item()
    # unique_values, unique_counts = torch.unique(x, return_counts=True)
    # res[unique_values.long()] = unique_counts
    return res


def limit_iterable(iterable, k):
    return itertools.islice(iterable, k)


def get_file(stem, root, digit_count=4):
    os.makedirs(root, exist_ok=True)
    files = [d for d in os.listdir(root) if re.match(fr"{stem}\d{{{digit_count}}}$", d)]
    numbers = sorted(set(int(re.search(r"\d+", d).group()) for d in files))
    mex = 0
    for num in numbers:
        if mex != num:
            break
        mex += 1

    filename = f"{stem}{mex:04}"
    with open(os.path.join(root, filename), "w") as _:
        pass
    return filename


def hadamard_transform_adaptive(A, scale):
    if A.is_cuda:
        return hadamard_transform(A, scale=scale)
    else:
        return F.linear(A, torch.tensor(scipy.linalg.hadamard(A.shape[-1]), dtype=torch.float32)) * scale


def random_k_mask(n, k, device):
    x = torch.zeros(n, device=device, dtype=torch.bool)
    indices = torch.randperm(n)[:k]
    x[indices] = True
    return x


def multiply_kronecker(x, n, mult_A, m, mult_B):
    # mult_A: x -> xA, mult_B: x -> xB
    # takes x -> x(A \otimes B)
    # A, B are square; n, m are their sizes
    x = rearrange(x, "k (n m) -> (k n) m", n=n)
    x = mult_B(x)
    x = rearrange(x, "(k n) m -> (k m) n", n=n)
    x = mult_A(x)
    x = rearrange(x, "(k m) n -> k (n m)", m=m)
    return x


def norm(X):
    return X.square().mean().sqrt()


def rmse(X, Y):
    return (X - Y).square().mean().sqrt()
