import torch
from torch.distributed import get_world_size, get_rank
import torch.distributed as dist
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
import os
from tqdm import tqdm
import argparse

from parallel.start import start
from parallel.ppl_utils import split_dataset, get_wikitext2
from parallel.config import no_q_config


def is_linear(module):
    return isinstance(module, ColumnParallelLinear) or \
           isinstance(module, RowParallelLinear)


class Hessian:
    def __init__(self, module, dst_rank):
        n = module.in_features
        self.dst_rank = dst_rank
        self.is_main = (dst_rank == get_rank())
        if self.is_main:
            self.H = torch.zeros((n, n), device="cuda", dtype=torch.float32)
        self.do_gather = isinstance(module, RowParallelLinear)
        if self.do_gather or self.is_main:
            module.register_forward_pre_hook(lambda _, input: self.update_on_input(input[0]))

    def update_on_input(self, F):
        if self.do_gather:
            if self.is_main:
                tensor_list = [torch.zeros_like(F) for i in range(get_world_size())]
                dist.gather(F, tensor_list, dst=self.dst_rank)
                F = torch.cat(tensor_list, dim=-1)
            else:
                dist.gather(F, None, dst=self.dst_rank)
                return

        assert self.is_main
        F = F.view(-1, F.shape[-1]).to(torch.float32)
        self.H.addmm_(F.T, F)

    def get(self):
        assert self.is_main
        return self.H


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--is-llama-2", choices=["True", "False"], required=True)
    parser.add_argument("--store-path", type=str, required=True,
                        help="Path to directory where hessians will be stored")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--L", type=int, help="Minimum layer ID to compute Hessian")
    parser.add_argument("--R", type=int, help="Maximum layer ID to compute Hessian")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ckpt_path = args.ckpt_path
    is_llama_2 = args.is_llama_2 == "True"
    seqlen = args.seqlen
    store_path = args.store_path
    L = args.L
    R = args.R

    model, tokenizer = start(ckpt_path, is_llama_2, no_q_config)
    world_size = get_world_size()

    wikitext = get_wikitext2(tokenizer=tokenizer, is_testset=False)
    wikitext = split_dataset(wikitext, seqlen)

    modules = sorted(
        [(name, module) for name, module in model.named_modules() \
                        if is_linear(module)], key=lambda x: x[0])
    hessians = {}
    print("Module count:", len(modules))
    for i, pp in enumerate(modules):
        if (L is None or i >= L) and (R is None or i < R):
            name, module = pp
            hessians[name] = Hessian(module, i % world_size)

    for i in tqdm(range(wikitext.shape[0])):
        batch = wikitext[i:i+1].to("cuda")
        model(batch, start_pos=0)
 
    os.makedirs(store_path, exist_ok=True)

    N = wikitext.shape[0]
    for layer_name, hessian in hessians.items():
        if hessian.is_main:
            torch.save(hessian.get() / N, os.path.join(store_path, layer_name))


if __name__ == "__main__":
    main()
