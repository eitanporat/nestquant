import torch
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt

from e8 import encode_e8, G, G_inv, generate_dither
from utils import matrix_to_pieces, pieces_to_matrix


# Set default dtype if you want (optional):
# torch.set_default_dtype(torch.float32)
def encode(x, q, z, beta):
    # x, z expected to be torch tensors
    # encode_e8 expects x: [N,8]
    x = x / beta
    if z is not None:
        x = x + z
    t = encode_e8(x)
    y = torch.round(t @ G_inv.T.to(t.device)).to(torch.int64)
    res = torch.remainder(y, q)
    if z is not None:
        t = t - z
    lambda_c = q * encode_e8(t / q)
    # err: sum over axis=1
    err = torch.sum(lambda_c * lambda_c, dim=-1) > 1e-9
    return res, err


def decode(enc, q, z, beta):
    # enc: int tensor
    # Convert to float
    encf = enc.float()
    y_tilde = encf @ G.T.to(encf.device)
    if z is not None:
        y_tilde = y_tilde - z
    x = beta * (y_tilde - q * encode_e8(y_tilde / q))
    return x


def encode_matrix_first_beta(A, q, betas, seed, use_dither=True):
    device = A.device
    pieces = matrix_to_pieces(A)
    rows = pieces.shape[0]
    beta_id = torch.full((rows,), -1, dtype=torch.int8, device=device)
    enc = torch.zeros(pieces.shape, dtype=torch.int64, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for i, beta_0 in enumerate(betas):
        beta = beta_0 / q
        mask = (beta_id == -1)
        pieces_to_quantize = pieces[mask]
        z = generate_dither(pieces_to_quantize.shape[0], generator, device) if use_dither else None
        cur_enc, err = encode(pieces_to_quantize, q, z, beta)
        masked_beta_id = beta_id[mask]
        masked_beta_id[~err] = i
        beta_id[mask] = masked_beta_id

        enc[mask] = cur_enc

    assert not (beta_id == -1).any()

    return {
        "shape": A.shape,
        "enc": enc,
        "beta_id": beta_id,
    }


def encode_matrix_try_all(A, q, betas, seed, use_dither=True):
    device = A.device
    pieces = matrix_to_pieces(A)
    rows = pieces.shape[0]
    beta_id = torch.full((rows,), 0, dtype=torch.int8, device=device)
    enc = torch.zeros(pieces.shape, dtype=torch.int64, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    INF = 1e9
    error = torch.full((rows,), INF, device=device, dtype=torch.float32)
    no_overflow = torch.full((rows,), False, dtype=torch.bool, device=device)

    for i, beta_0 in enumerate(betas):
        beta = beta_0 / q
        z = generate_dither(pieces.shape[0], generator, device) if use_dither else None
        cur_enc, err = encode(pieces, q, z, beta)
        recon = decode(cur_enc, q, z, beta)
        no_overflow[~err] = True
        cur_error = ((recon - pieces) ** 2).sum(dim=1)
        replace_mask = cur_error < error
        enc[replace_mask] = cur_enc[replace_mask]
        beta_id[replace_mask] = i
        error[replace_mask] = cur_error[replace_mask]

    # assert no_overflow.all()

    return {
        "shape": A.shape,
        "enc": enc,
        "beta_id": beta_id,
    }


def decode_matrix_first_beta(A_enc, q, betas, seed, use_dither=True):
    enc = A_enc["enc"]
    device = enc.device
    result_pieces = torch.zeros(enc.shape, device=device, dtype=torch.float32)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for i, beta_0 in enumerate(betas):
        beta = beta_0 / q
        mask_ge = A_enc["beta_id"] >= i
        mask = A_enc["beta_id"] == i
        rel_mask = mask[mask_ge]

        # Tricky: when we generated dither, we didn't know which entries would be actually used
        z = generate_dither(mask_ge.sum().item(), generator, device)[rel_mask] if use_dither else None
        cur = enc[mask]

        cur_dec = decode(cur, q, z, beta)
        result_pieces[mask] = cur_dec
    return pieces_to_matrix(result_pieces, A_enc["shape"])


def decode_matrix_try_all(A_enc, q, betas, seed, use_dither=True):
    enc = A_enc["enc"]
    device = enc.device
    result_pieces = torch.zeros(enc.shape, device=device, dtype=torch.float32)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for i, beta_0 in enumerate(betas):
        beta = beta_0 / q
        mask = A_enc["beta_id"] == i
        z = generate_dither(enc.shape[0], generator, device)[mask] if use_dither else None
        cur = enc[mask]
        recon = decode(cur, q, z, beta)
        result_pieces[mask] = recon
    return pieces_to_matrix(result_pieces, A_enc["shape"])


# copied from QuIP# paper
def block_LDL(H, b, check_nan=True):
    n = H.shape[0]
    assert (n % b == 0)
    m = n // b
    try:
        L = torch.linalg.cholesky(H)
    except:
        return None
    DL = torch.diagonal(L.reshape(m, b, m, b), dim1=0, dim2=2).permute(2, 0, 1)
    D = (DL @ DL.permute(0, 2, 1)).cpu()
    DL = torch.linalg.inv(DL)
    L = L.view(n, m, b)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]

    if check_nan and L.isnan().any():
        return None

    L = L.reshape(n, n)
    return (L, D.to(DL.device))


def encode_matrix_LDLQ_first_beta(A, H, q, betas, seed, use_dither=True):
    # A -- shape (m, n)
    # H -- shape (n, n); H[i, j] is (d obj) / dH_{ki}H_{kj} forall k
    m, n = A.shape
    device = A.device
    assert n % 8 == 0

    # A_hat - A
    L, D = block_LDL(H, 8)

    beta_id = torch.full((m, n // 8), -1, dtype=torch.int8, device=device)
    enc = torch.zeros((m, n), dtype=torch.int64, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    decoded = A.clone()
    new_A = A.clone()

    for c in range(n - 8, -8, -8):
        block_idx = c // 8
        # Quantize column block c:c+8
        WXWX = A[:, c:c+8] + (A[:, c+8:n] - decoded[:, c+8:n]) @ L[c+8:n, c:c+8]
        decoded[:, c:c+8] = WXWX
        new_A[:, c:c+8] = WXWX
        for i, beta_0 in enumerate(betas):
            beta = beta_0 / q
            mask = (beta_id[:, block_idx] == -1)
            rows_to_quantize = WXWX[mask]
            z = generate_dither(rows_to_quantize.shape[0], generator, device) if use_dither else None
            cur_enc, err = encode(rows_to_quantize, q, z, beta)
            masked_beta_id = beta_id[mask, block_idx]
            masked_beta_id[~err] = i
            beta_id[mask, block_idx] = masked_beta_id
            enc[mask, c:c+8] = cur_enc
            decoded[mask, c:c+8] = decode(cur_enc, q, z, beta)

    if (beta_id == -1).any():
        bad_mask = (beta_id == -1)
        pos = torch.nonzero(bad_mask)[0].tolist()
        print("Bad vector piece")
        print("pos:", pos[0], pos[1])
        print(new_A[pos[0], pos[1] * 8: (pos[1] + 1) * 8])
        assert False
    return {
        "enc": enc,
        "beta_id": beta_id,
    }


def decode_matrix_LDLQ_first_beta(A_enc, q, betas, seed, use_dither=True):
    enc = A_enc["enc"]
    beta_id = A_enc["beta_id"]
    device = enc.device
    result = torch.zeros(enc.shape, device=device, dtype=torch.float32)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    (m, n) = enc.shape

    for c in range(n-8, -8, -8):
        block_idx = c // 8
        for i, beta_0 in enumerate(betas):
            beta = beta_0 / q
            mask_ge = beta_id[:, block_idx] >= i
            mask = beta_id[:, block_idx] == i
            rel_mask = mask[mask_ge]

            # Tricky: when we generated dither, we didn't know which entries would be actually used
            z = generate_dither(mask_ge.sum().item(), generator, device)[rel_mask] if use_dither else None
            cur = enc[mask, c:c+8]

            cur_dec = decode(cur, q, z, beta)
            result[mask, c:c+8] = cur_dec
    return result

# from torch.distributed import get_rank

def encode_matrix_LDLQ_try_all(A, H, q, betas, seed, use_dither=True):
    # A -- shape (m, n)
    # H -- shape (n, n); H[i, j] is (d obj) / dH_{ki}H_{kj} forall k
    # print("rank:", get_rank(), A.device, H.device)
    m, n = A.shape
    device = A.device
    assert n % 8 == 0

    # A_hat - A
    result = block_LDL(H, 8)
    if result is None:
        print("LDL decomposition of H does not exist")
        H = torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        result = block_LDL(H, 8)
        assert result is not None
    L, D = result

    beta_id = torch.full((m, n // 8), 0, dtype=torch.int8, device=device)
    enc = torch.zeros((m, n), dtype=torch.int64, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    decoded = A.clone()

    for c in range(n - 8, -8, -8):
        block_idx = c // 8
        # Quantize column block c:c+8
        WXWX = A[:, c:c+8] + (A[:, c+8:n] - decoded[:, c+8:n]) @ L[c+8:n, c:c+8]
        decoded[:, c:c+8] = WXWX

        INF = 1e9
        error = torch.full((A.shape[0],), INF, device=device, dtype=torch.float32)
        no_overflow = torch.full((A.shape[0],), False, dtype=torch.bool, device=device)

        for i, beta_0 in enumerate(betas):
            beta = beta_0 / q
            z = generate_dither(WXWX.shape[0], generator, device) if use_dither else None
            cur_enc, err = encode(WXWX, q, z, beta)
            recon = decode(cur_enc, q, z, beta)
            no_overflow[~err] = True
            cur_error = ((recon - WXWX) ** 2).sum(dim=1)
            replace_mask = cur_error < error
            enc[replace_mask, c:c+8] = cur_enc[replace_mask]
            beta_id[replace_mask, block_idx] = i
            error[replace_mask] = cur_error[replace_mask]
            decoded[replace_mask, c:c+8] = recon[replace_mask]
        if not no_overflow.all():
            print("Warning, overflow")

    if (beta_id == -1).any():
        print("This should not be possible, given the previous assert")
        assert False
    return {
        "enc": enc,
        "beta_id": beta_id,
    }


def decode_matrix_LDLQ_try_all(A_enc, q, betas, seed, use_dither=True):
    enc = A_enc["enc"]
    beta_id = A_enc["beta_id"]
    device = enc.device
    result = torch.zeros(enc.shape, device=device, dtype=torch.float32)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    (m, n) = enc.shape

    for c in range(n-8, -8, -8):
        block_idx = c // 8
        for i, beta_0 in enumerate(betas):
            beta = beta_0 / q
            mask = beta_id[:, block_idx] == i
            z = generate_dither(enc.shape[0], generator, device)[mask] if use_dither else None
            cur = enc[mask, c:c+8]
            cur_dec = decode(cur, q, z, beta)
            result[mask, c:c+8] = cur_dec

    return result


def encode_matrix(A, q, betas, seed, use_dither=True, try_all=False, H=None):
    if H is None:
        if try_all:
            return encode_matrix_try_all(A, q, betas, seed, use_dither=use_dither)
        else:
            return encode_matrix_first_beta(A, q, betas, seed, use_dither=use_dither)
    else:
        if try_all:
            return encode_matrix_LDLQ_try_all(A, H, q, betas, seed, use_dither=use_dither)
        else:
            return encode_matrix_LDLQ_first_beta(A, H, q, betas, seed, use_dither=use_dither)



def decode_matrix(A, q, betas, seed, use_dither=True, try_all=False, use_ldlq=False):
    if not use_ldlq:
        if try_all:
            return decode_matrix_try_all(A, q, betas, seed, use_dither=use_dither)
        else:
            return decode_matrix_first_beta(A, q, betas, seed, use_dither=use_dither)
    else:
        if try_all:
            return decode_matrix_LDLQ_try_all(A, q, betas, seed, use_dither=use_dither)
        else:
            return decode_matrix_LDLQ_first_beta(A, q, betas, seed, use_dither=use_dither)
