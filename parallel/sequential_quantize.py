import argparse
import os
import copy
import logging
from datetime import datetime

import torch
import torch.distributed as dist
from tqdm import tqdm
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear

# Ensure high precision for matmul
torch.set_float32_matmul_precision("high")

from parallel.start import start
from parallel.config import no_q_config
from parallel.ppl_utils import split_dataset, get_wikitext2
from config import create_config
from quant_utils import quantsim, quantsim_col, rot_hess
from hadamard import kron_h_ip

# --- Setup logging: file + plain console handlers ---
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"quantize_{timestamp}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler with timestamps
file_h = logging.FileHandler(log_path)
file_h.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
file_h.setFormatter(file_formatter)

# Console handler plain
console_h = logging.StreamHandler()
console_h.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_h.setFormatter(console_formatter)

logger.addHandler(file_h)
logger.addHandler(console_h)

# Weight ordering for sorting modules
weight_order = ["wk", "wv", "wq", "wo", "w1", "w3", "w2"]
prio = {suffix: idx for idx, suffix in enumerate(weight_order)}
DEFAULT = len(prio)
INF = float("inf")

def is_linear(module):
    return isinstance(module, (ColumnParallelLinear, RowParallelLinear))

def is_col(weight_name: str) -> bool:
    return "w2" in weight_name or "wo" in weight_name

def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

def module_sort_key(item):
    name, _ = item
    if name == "output" or name.startswith("output."):
        group = 2
    elif name.startswith("layers."):
        group = 0
    else:
        group = 1
    layer_idx = int(name.split(".")[1]) if group == 0 else INF
    suffix = name.split(".")[-1]
    suffix_prio = prio.get(suffix, DEFAULT)
    return (group, layer_idx, suffix_prio, name)

def shared_input_key(name: str) -> str:
    if ".attention.w" in name and not ".attention.wo" in name:
        return name.split(".attention.")[0] + ".attention"
    if ".feed_forward.w1" in name or ".feed_forward.w3" in name:
        return name.split(".feed_forward.")[0] + ".feed_forward_w13"
    return name

class NoiseCovariance:
    def __init__(self, clean_module, quant_module, dst_rank, *, keep_on_gpu=True):
        in_features = clean_module.in_features
        self.dst_rank = dst_rank
        self.is_main = (dst_rank == get_rank())
        self.keep_on_gpu = keep_on_gpu
        device = "cuda" if keep_on_gpu else "cpu"
        if self.is_main:
            self.J = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
        self.do_gather = isinstance(clean_module, RowParallelLinear)
        self._clean_input = None
        self._clean_hook_handle = clean_module.register_forward_pre_hook(self._hook_clean)
        self._quant_hook_handle = quant_module.register_forward_pre_hook(self._hook_quant)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False

    def _hook_clean(self, _module, inputs):
        self._clean_input = inputs[0].detach()

    def _hook_quant(self, _module, inputs):
        if self._clean_input is None:
            raise RuntimeError("Clean model pass must precede quantized pass.")
        noise = inputs[0] - self._clean_input
        self.update_on_input(noise)
        self._clean_input = None

    def update_on_input(self, Z):
        if self.do_gather:
            ws = get_world_size()
            if self.is_main:
                tl = [torch.empty_like(Z, device=Z.device) for _ in range(ws)]
                dist.gather(Z, tl, dst=self.dst_rank)
                Z = torch.cat(tl, dim=-1)
            else:
                dist.gather(Z, [], dst=self.dst_rank)
                return
        if not self.is_main:
            return
        Z = Z.view(-1, Z.shape[-1]).to(torch.float32)
        if not self.keep_on_gpu:
            Z = Z.cpu()
        self.J.addmm_(Z.T, Z)

    def get(self):
        assert self.is_main
        return self.J.cpu()

    def remove(self):
        self._clean_hook_handle.remove()
        self._quant_hook_handle.remove()
        if hasattr(self, "J"):
            del self.J
        if hasattr(self, "_clean_input"):
            del self._clean_input

def parse_arguments():
    p = argparse.ArgumentParser(description="Efficient sequential quantization with shared input noise caching.")
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--is-llama-2", choices=["True", "False"], required=True)
    p.add_argument("--store-path", required=True)
    p.add_argument("--hess-path", required=True, help="Path to pre-computed clean Hessians (H).")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--L", type=int, help="Start quantization at this layer index.")
    p.add_argument("--R", type=int, help="End quantization before this layer index.")
    p.add_argument("--q", type=int, default=14)
    p.add_argument("--quant-act", action="store_true", help="Enable activation quantization.")
    p.add_argument("--quant-kv", action="store_true", help="Enable K/V cache quantization.")
    p.add_argument("--act-betas", nargs="+", type=float, default=[3.47, 4.74, 6.90, 18.11])
    p.add_argument("--key-betas", nargs="+", type=float, default=[3.50, 4.58, 6.47, 17.06])
    p.add_argument("--value-betas", nargs="+", type=float, default=[3.53, 5.59, 9.62, 29.03])
    p.add_argument("--use-scalar", action="store_true", help="Use scalar covariance for J.")
    p.add_argument("--use-diagonal", action="store_true", help="Use only the diagonal of J for quantization.")
    return p.parse_args()

def main():
    args = parse_arguments()
    # Pretty-print args on separate lines
    logger.info("Running with arguments:")
    for key, val in vars(args).items():
        logger.info("  %s: %s", key, val)

    # Initialize models and data
    model, tokenizer = start(args.ckpt_path, args.is_llama_2 == "True", no_q_config)
    clean_model = copy.deepcopy(model).cuda().eval()
    quantized_model = model.cuda().eval()

    wikitext = split_dataset(get_wikitext2(tokenizer=tokenizer, is_testset=False), args.seqlen)
    modules = sorted(
        [(n, m) for n, m in quantized_model.named_modules() if is_linear(m)],
        key=module_sort_key,
    )

    qconfig = create_config(
        q=args.q,
        quant_act=args.quant_act,
        quant_kv=args.quant_kv,
        act_betas=args.act_betas,
        key_betas=args.key_betas,
        value_betas=args.value_betas,
    )

    world_size = get_world_size()
    batch_size = 8
    total = wikitext.shape[0]

    # Cache for noise covariances by shared input key
    group_J = {}

    for idx, (name, quant_module) in enumerate(modules):
        if args.L is not None and idx < args.L:
            continue
        if args.R is not None and idx >= args.R:
            continue

        logger.info("--- Quantizing layer %d/%d: %s ---", idx+1, len(modules), name)

        gkey = shared_input_key(name)
        if gkey not in group_J:
            logger.info("First encounter of input group '%s'. Computing noise covariance J.", gkey)
            needs_noise_calc = (idx > 0) and (args.quant_act or args.quant_kv)
            J = None
            if needs_noise_calc:
                clean_module = clean_model.get_submodule(name)
                dst_rank_for_j = idx % max(world_size, 1)
                with NoiseCovariance(clean_module, quant_module, dst_rank=dst_rank_for_j, keep_on_gpu=True) as noise_calc:
                    for i in tqdm(range(0, total, batch_size), desc=f"J[{gkey}]", leave=False):
                        batch = wikitext[i:i+batch_size].cuda()
                        with torch.no_grad():
                            clean_model(batch, start_pos=0)
                            quantized_model(batch, start_pos=0)
                    if noise_calc.is_main:
                        J = noise_calc.get() / wikitext.numel()
                        # Apply scalar or diagonal if requested
                        if args.use_scalar:
                            mean_diag = J.diag().mean()
                            J = torch.eye(J.size(0), dtype=J.dtype, device=J.device) * mean_diag
                        if args.use_diagonal:
                            J = torch.diag(J.diag())
                # Broadcast J across ranks
                if world_size > 1:
                    shape_tensor = torch.tensor(J.shape, device='cuda', dtype=torch.long)
                    dist.broadcast(shape_tensor, src=dst_rank_for_j)
                    if get_rank() != dst_rank_for_j:
                        J = torch.zeros(shape_tensor.tolist(), dtype=torch.float32)
                    J = J.cuda()
                    dist.broadcast(J, src=dst_rank_for_j)
            if J is None:
                J = torch.zeros((quant_module.in_features, quant_module.in_features), dtype=torch.float32)
                if args.use_diagonal:
                    J = torch.diag(J.diag())
            group_J[gkey] = J
        else:
            logger.info("Re-using cached noise covariance J for group '%s'.", gkey)

        # Retrieve J
        J = group_J[gkey].cuda() if torch.cuda.is_available() else group_J[gkey]
        clean_H = torch.load(os.path.join(args.hess_path, name), map_location='cuda').float()

        betas = (args.act_betas if "act" in name else
                 args.key_betas if "key" in name else
                 args.value_betas)
        j_norm = torch.mean(J ** 2).item()
        h_norm = torch.mean(clean_H ** 2).item()
        use_j = (j_norm / (h_norm + 1e-20)) > 1e-8

        logger.info("First 10 diag(H): %s", clean_H.diag()[:10])
        logger.info("First 10 diag(J): %s", J.diag()[:10])
        logger.info("H norm: %.4e, J norm: %.4e, trace(J): %.4e, trace(H): %.4e use_j=%s", h_norm, j_norm, J.diag().mean().item(), clean_H.diag().mean().item(), use_j)

        q_func = quantsim_col if is_col(name) else quantsim
        q_weight = q_func(quant_module.weight, args.q, betas,
                          rot=kron_h_ip, H=clean_H, J=(J if use_j else None))
        with torch.no_grad():
            quant_module.weight.copy_(q_weight.to(quant_module.weight.dtype))
        quant_module.qconfig = qconfig
        torch.cuda.empty_cache()

    os.makedirs(args.store_path, exist_ok=True)
    if get_rank() == 0:
        logger.info("Saving quantized model state dict to %s", args.store_path)
        torch.save(quantized_model.state_dict(), os.path.join(args.store_path, "quantized_model.pth"))

if __name__ == '__main__':
    main()