from parallel.start import start
from parallel.ppl_utils import get_wikitext2, compute_perplexity
from parallel.config import create_config
from shutil import copy
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_act", action="store_true",
                        help="Flag to quantize activations")
    parser.add_argument("--quant_kv", action="store_true",
                        help="Flag to quantize KV cache")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Path to quantized model directory")
    parser.add_argument("--is-llama-2", choices=["True", "False"], required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--act_betas", type=float, nargs='+', default=[3.47, 4.74, 6.90, 18.11],
                        help="Betas for activations")
    parser.add_argument("--key_betas", type=float, nargs='+', default=[3.50, 4.58, 6.47, 17.06],
                        help="Betas for keys")
    parser.add_argument("--value_betas", type=float, nargs='+', default=[3.53, 5.59, 9.62, 29.03],
                        help="Betas for values")
    
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = args.ckpt_dir
    is_llama_2 = args.is_llama_2 == "True"
    seqlen = args.seqlen

    q = args.q
    quant_act = args.quant_act
    quant_kv = args.quant_kv
    act_betas = args.act_betas
    key_betas = args.key_betas
    value_betas = args.value_betas

    qconfig = create_config(q, quant_act, quant_kv, act_betas, key_betas, value_betas)

    model, tokenizer = start(ckpt_dir, is_llama_2, qconfig)
    wikitext = get_wikitext2(tokenizer=tokenizer, is_testset=True)
    ppl = compute_perplexity(model, wikitext, seqlen)
    print("PPL:", ppl)


if __name__ == "__main__":
    main()
