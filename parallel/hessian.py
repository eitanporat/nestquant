import os
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.distributed import get_world_size, get_rank

from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from parallel.start import start                 
from parallel.ppl_utils import split_dataset, get_wikitext2
from parallel.config import no_q_config


torch.set_float32_matmul_precision("high")

def is_linear(module):
    return isinstance(module, ColumnParallelLinear) or \
           isinstance(module, RowParallelLinear)


class Hessian:
    def __init__(self, module, dst_rank, *, keep_on_gpu=True):
        in_features     = module.in_features
        self.dst_rank   = dst_rank
        self.is_main    = (dst_rank == get_rank())
        print(get_rank(), dst_rank)
        self.keep_on_gpu = keep_on_gpu

        device = "cuda" if keep_on_gpu else "cpu"
        if self.is_main:
            self.H = torch.zeros((in_features, in_features),
                                 dtype=torch.float32,
                                 device=device)

        self.do_gather = isinstance(module, RowParallelLinear)
        self._handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        self.update_on_input(inputs[0])

    def update_on_input(self, F):
        if self.do_gather:
            if self.is_main:
                ws = get_world_size()
                tl = [torch.zeros_like(F) for _ in range(ws)]
                dist.gather(F, tl, dst=self.dst_rank)
                F = torch.cat(tl, dim=-1)
            else:
                dist.gather(F, None, dst=self.dst_rank)
                return                        

        if not self.is_main:
            return                           

        F = F.view(-1, F.shape[-1]).to(torch.float32)   
        if not self.keep_on_gpu:
            F = F.cpu()
        self.H.addmm_(F.T, F)                             

    def get(self):
        assert self.is_main
        return self.H.cpu()

    def remove(self):
        self._handle.remove()
        if hasattr(self, "H"):
            del self.H


@torch.inference_mode()     
def run_in_chunks(
        model,
        modules,
        wikitext,
        *,
        world_size: int,
        chunk_size: int,
        store_path: str,
        keep_on_gpu: bool = True,
):
    os.makedirs(store_path, exist_ok=True)
    N = wikitext.shape[0]                

    total_layers = len(modules)
        
    for chunk_start in range(0, total_layers, chunk_size):
        hessians = {}
        for local_idx, (name, module) in enumerate(
                modules[chunk_start:chunk_start + chunk_size]):
            global_idx = chunk_start + local_idx
            hessians[name] = Hessian(module,
                                     dst_rank=global_idx % world_size,
                                     keep_on_gpu=keep_on_gpu)

        for i in tqdm(range(N), leave=False,
                      desc=f"Layers {chunk_start}-{min(chunk_start+chunk_size-1, total_layers-1)}"):
            batch = wikitext[i:i + 1].to("cuda", non_blocking=True)
            model(batch, start_pos=0)         

        for layer_name, h in hessians.items():
            if h.is_main:
                torch.save(h.get() / wikitext.numel(), os.path.join(store_path, layer_name))
            h.remove()

        torch.cuda.empty_cache()               


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path",   type=str, required=True,
                   help="Path to model checkpoint directory")
    p.add_argument("--is-llama-2",  choices=["True", "False"], required=True)
    p.add_argument("--store-path",  type=str, required=True,
                   help="Directory in which to save Hessian tensors")
    p.add_argument("--seqlen",      type=int, default=2048,
                   help="Sequence length used to chop WikiText-2")
    p.add_argument("--L",           type=int,
                   help="Compute Hessians from this (inclusive) layer id")
    p.add_argument("--R",           type=int,
                   help="Up-to-but-NOT-including this layer id")
    p.add_argument("--chunk-size",  type=int, default=64,
                   help="#layers to keep resident at once")
    p.add_argument("--cpu-accum",   action="store_true",
                   help="Accumulate H on CPU (reduces GPU RAM, slower)")
    return p.parse_args()


def main():
    args        = parse_arguments()
    ckpt_path   = args.ckpt_path
    is_llama_2  = args.is_llama_2 == "True"
    seqlen      = args.seqlen
    store_path  = args.store_path
    L, R        = args.L, args.R
    chunk_size  = args.chunk_size
    keep_gpu    = not args.cpu_accum

    model, tokenizer = start(ckpt_path, is_llama_2, no_q_config)
    model.eval()

    world_size = get_world_size()

    wikitext = get_wikitext2(tokenizer=tokenizer, is_testset=False)
    wikitext = split_dataset(wikitext, seqlen)       

    modules = [(name, module) for name, module in model.named_modules()
                        if is_linear(module)]
    
    modules = sorted(modules, key=lambda x: x[0])

    if L is not None or R is not None:
        modules = modules[L if L is not None else 0 :
                          R if R is not None else len(modules)]

    print(f"Total linear layers considered: {len(modules)}")

    run_in_chunks(model,
                  modules,
                  wikitext,
                  world_size=world_size,
                  chunk_size=chunk_size,
                  store_path=store_path,
                  keep_on_gpu=keep_gpu)


if __name__ == "__main__":
    main()