from hessian import Hessian, is_linear
from parallel.start import start
from hadamard import kron_h_ip
from parallel.config import no_q_config

# in sequential quantize 
# we compute the hessian of the first layer and the hessian of the noise

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

    parser.add_argument("--q", type=int, default=14)
    parser.add_argument("--act_betas", type=float, nargs='+', default=[3.47, 4.74, 6.90, 18.11],
                        help="Betas for activations")
    parser.add_argument("--key_betas", type=float, nargs='+', default=[3.50, 4.58, 6.47, 17.06],
                        help="Betas for keys")
    parser.add_argument("--value_betas", type=float, nargs='+', default=[3.53, 5.59, 9.62, 29.03],
                        help="Betas for values")

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

    q = args.q
    quant_act = args.quant_act
    quant_kv = args.quant_kv
    act_betas = args.act_betas
    key_betas = args.key_betas
    value_betas = args.value_betas

    qconfig = create_config(q, quant_act, quant_kv, act_betas, key_betas, value_betas)
    
    J = None
    
    for i, pp in enumerate(modules):
        if L is not None and i < L:
            continue
        if R is not None and i >= R:
            continue
                
        H = Hessian(module, 0)
        
        for i in tqdm(range(wikitext.shape[0])):
            batch = wikitext[i:i+1].to("cuda")
            model(batch, start_pos=0)

        # get name of module
        # name_nw 
        clean_H = torch.load(os.path.join(hess_path, name_nw), map_location="cuda", weights_only=True).float()
        J = H - clean_H
        
        if is_col(name):
            module.weight = quantsim_col(module.weight, q, betas, rot=kron_h_ip, H=H, J=J)
        else:
            module.weight = quantsim(module.weight, q, betas, rot=kron_h_ip, H=H, J=J)

        module.qconfig = qconfig
        
    # save state_dict
    torch.save(model.state_dict(), os.path.join(store_path, "quantized_model.pth"))