import argparse
import os

from tqdm import tqdm
import torch
import torch.distributed as dist

torch.set_float32_matmul_precision("high")

from hessian import Hessian as _Hessian, is_linear
from parallel.start import start
from parallel.config import no_q_config
from parallel.ppl_utils import split_dataset, get_wikitext2
from config import create_config
from quant_utils import quantsim, quantsim_col
from hadamard import kron_h_ip

weight_order = ["wk", "wv", "wq", "wo", "w1", "w3", "w2"]
prio = {suffix: idx for idx, suffix in enumerate(weight_order)}
DEFAULT = len(prio)  
INF      = float("inf")


def is_col(weight_name: str) -> bool:
    return "w2" in weight_name or "wo" in weight_name


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

    if group == 0:
        layer_idx = int(name.split(".")[1])
    else:
        layer_idx = INF               

    suffix = name.split(".")[-1]
    suffix_prio = prio.get(suffix, DEFAULT)

    return (group, layer_idx, suffix_prio, name)


class Hessian(_Hessian):
    def __init__(self, module, dst_rank, *, keep_on_gpu=True):
        super().__init__(module, dst_rank, keep_on_gpu=keep_on_gpu)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False


def shared_input_key(name: str) -> str:
    if ".attention.w" in name and not ".attention.wo" in name:
        return name.split(".attention.")[0] + ".attention"
    if ".feed_forward.w1" in name or ".feed_forward.w3" in name:
        return name.split(".feed_forward.")[0] + ".feed_forward_w13"
    return name


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--is-llama-2", choices=["True", "False"], required=True)
    p.add_argument("--store-path", required=True)
    p.add_argument("--hess-path", required=True)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--L", type=int)
    p.add_argument("--R", type=int)
    p.add_argument("--q", type=int, default=14)
    p.add_argument("--quant-act", action="store_true")
    p.add_argument("--quant-kv", action="store_true")
    p.add_argument("--act-betas", nargs="+", type=float,
                   default=[3.47, 4.74, 6.90, 18.11])
    p.add_argument("--key-betas", nargs="+", type=float,
                   default=[3.50, 4.58, 6.47, 17.06])
    p.add_argument("--value-betas", nargs="+", type=float,
                   default=[3.53, 5.59, 9.62, 29.03])
    return p.parse_args()


def main():
    args = parse_arguments()
    model, tokenizer = start(args.ckpt_path,
                             args.is_llama_2 == "True",
                             no_q_config)
    model.cuda().eval()
    wikitext = split_dataset(
        get_wikitext2(tokenizer=tokenizer, is_testset=False),
        args.seqlen
    )

    modules = sorted(
        [(n, m) for n, m in model.named_modules() if is_linear(m)],
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
    batch_size = 32
    total = wikitext.shape[0]

    group_H: dict[str, torch.Tensor] = {}
    N = wikitext.shape[0]                

    
    for idx, (name, module) in enumerate(modules):
        if args.L is not None and idx < args.L:
            continue
        if args.R is not None and idx >= args.R:
            continue

        gkey = shared_input_key(name)
        if gkey not in group_H:
            with Hessian(module,
                         dst_rank=idx % max(world_size, 1),
                         keep_on_gpu=True) as H:
                for i in tqdm(range(0, total, batch_size),
                              desc=f"H[{gkey}]", leave=False):
                    batch = wikitext[i : i + batch_size].cuda()
                    model(batch, start_pos=0)
                H_obs = H.get() / wikitext.numel()
            group_H[gkey] = H_obs

        H_obs = group_H[gkey].cuda()
        clean_H = torch.load(
            os.path.join(args.hess_path, f"{name}"),
            map_location="cuda",
            weights_only=True,
        ).float()

        print(f"H_obs mean squared: {(H_obs ** 2).mean().item()} clean_H mean squared: {(clean_H ** 2).mean().item()}")
        J = H_obs - clean_H
        
        J = J * 1000
        clean_H = clean_H * 1000
        # numerical stability
        
        # print J mean squared message
        print(f"J[{name}] mean squared: {torch.mean(J**2).item()}")
        # print clean_H mean squared message
        print(f"clean_H[{name}] mean squared: {torch.mean(clean_H**2).item()}")
        
        
        betas = (
            args.act_betas if "act" in name
            else args.key_betas if "key" in name
            else args.value_betas
        )
        
        big_err = (J**2).mean() / (clean_H**2).mean() 
        
        if is_col(name):
            q_weight = quantsim_col(
                module.weight,
                args.q,
                betas,
                rot=kron_h_ip,
                H=clean_H,
                J=J if big_err > 1e-6 else None,
            )
        else:
            q_weight = quantsim(
                module.weight,
                args.q,
                betas,
                rot=kron_h_ip,
                H=clean_H,
                J=J if big_err > 1e-6 else None,
            )

        q_weight = q_weight.to(module.weight.dtype)  
        print("Weight mean squared", ((module.weight) ** 2).mean().item(), "Quantized weight mean squared", (q_weight ** 2).mean().item())
        with torch.no_grad():
            module.weight.copy_(q_weight)                  

        module.qconfig = qconfig
        torch.cuda.empty_cache()

    os.makedirs(args.store_path, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(args.store_path, "quantized_model.pth"))


if __name__ == "__main__":
    main()