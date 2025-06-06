from torch.distributed import get_world_size, get_rank
import torch
import torch.distributed as dist
import yaml
import argparse
from tqdm import tqdm

from parallel.start import start
from parallel.ppl_utils import get_wikitext2, split_dataset
from parallel.hessian import is_linear
from parallel.config import no_q_config


class Wrapper:
    def __init__(self, module):
        self.do_gather = isinstance(module, RowParallelLinear)
        self.items = 0
        self.sum_diff2 = 0.0
        self.sum2 = 0.0
        if get_rank() == 0 or self.do_gather:
            module.register_forward_pre_hook(lambda _, input: self.update_on_input(input[0]))

    def update_on_input(self, F):
        if self.do_gather:
            if get_rank() == 0:
                tensor_list = [torch.zeros_like(F) for i in range(get_world_size())]
                dist.gather(F, tensor_list, dst=0)
                F = torch.cat(tensor_list, dim=-1)
            else:
                dist.gather(F, None, dst=0)
                return

        assert get_rank() == 0
        F = F.view(-1, F.shape[-1]).to(torch.float32)
        self.items += torch.numel(F)
        self.sum2 += torch.sum(F ** 2).item()

    def get(self):
        return {
            "x_var": self.sum2 / self.items,
        }


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--is-llama-2", choices=["True", "False"], required=True)
    parser.add_argument("--store-path", type=str, required=True,
                        help="Path to resulting yaml")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of wikitext2 seqeunces")
    parser.add_argument("--seqlen", type=int, default=2048)
    return parser.parse_args()


def main():
    args = parse_arguments()
    ckpt_path = args.ckpt_path
    is_llama_2 = args.is_llama_2 == "True"
    seqlen = args.seqlen
    store_path = args.store_path
    samples = args.samples

    model, tokenizer = start(ckpt_path, is_llama_2, no_q_config)

    wikitext = get_wikitext2(tokenizer, is_testset=False)
    wikitext = split_dataset(wikitext, seqlen)
    wikitext = wikitext[:samples]

    modules = sorted(
        [(name, module) for name, module in model.named_modules() \
                        if is_linear(module)], key=lambda x: x[0])
    wrappers = {}
    print("Module count:", len(modules))
    for i, pp in enumerate(modules):
        name, module = pp
        wrappers[name] = Wrapper(module)

    for i in tqdm(range(wikitext.shape[0])):
        batch = wikitext[i:i+1].to("cuda")
        model(batch, start_pos=0)

    if get_rank() == 0:
        result = {}
        for layer_name, wrapper in wrappers.items():
            result[layer_name] = wrapper.get()
        with open(store_path, "w") as f:
            yaml.dump(result, f, default_flow_style=False)


if __name__ == "__main__":
    main()
