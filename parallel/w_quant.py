import torch
import yaml
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm
import os
import sys
import numpy as np
import argparse
import shutil

from parallel.quant_utils import quantsim, quantsim_col
from hadamard import kron_h_ip


def is_linear(weight_name):
    return "layers" in weight_name and "norm" not in weight_name


def is_col(weight_name):
    return "w2" in weight_name or "wo" in weight_name


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Path to unquantized model directory")
    parser.add_argument("--hess-path", type=str, required=True,
                        help="Path to the directory with hessians")
    parser.add_argument("--target-path", type=str, required=True,
                        help="Path to directory to save the model")
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--betas", type=float, nargs='+', default=[4.0, 5.5, 7.0, 18.0])
    parser.add_argument("--eps_cons", type=float, default=155.0,
                        help="Constant which estimates |x|^2/|q(x)-x|^2")
    parser.add_argument("--eps_path", type=str,
                        help="Path to yaml with magnitudes for qa-ldlq (optional)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ckpt_dir = args.ckpt_dir
    hess_path = args.hess_path
    target_path = args.target_path
    q = args.q
    betas = args.betas
    eps_cons = args.eps_cons
    eps_path = args.eps_path

    use_qa = eps_path is not None
    if use_qa:
        assert eps_cons is not None
        with open(eps_path) as f:
            eps_file = yaml.safe_load(f)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    local_rank = get_rank()
    torch.cuda.set_device(local_rank)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    checkpoints = list(sorted(filter(lambda x: x.endswith(".pth"), os.listdir(ckpt_dir))))
    assert len(checkpoints) == get_world_size()

    ckpt_path = os.path.join(ckpt_dir, checkpoints[local_rank])
    state_dict = torch.load(ckpt_path, map_location="cuda", weights_only=True)

    to_quantize = {name: weight for name, weight in state_dict.items() if is_linear(name)}

    os.makedirs(target_path, exist_ok=True)

    for i, name in tqdm(enumerate(list(sorted(to_quantize.keys())))):
        print(f"Quantizing {name}")
        name_nw = name.rsplit(".", 1)[0]
        H = torch.load(os.path.join(hess_path, name_nw), map_location="cuda", weights_only=True).float()
        eps = np.sqrt(eps_file[name_nw]["x_var"] / eps_cons) if use_qa else None
        X = state_dict[name].float()
        if is_col(name):
            X = quantsim_col(X, q, betas, rot=kron_h_ip, H=H, eps=eps)
        else:
            X = quantsim(X, q, betas, rot=kron_h_ip, H=H, eps=eps)
        X = X.bfloat16()
        state_dict[name] = X

    torch.save(state_dict, os.path.join(target_path, f"consolidated.{local_rank:02}.pth"))
    for filename in ["tokenizer.model", "params.json"]:
        shutil.copy(os.path.join(ckpt_dir, filename),
                    os.path.join(target_path, filename))


if __name__ == "__main__":
    main()
