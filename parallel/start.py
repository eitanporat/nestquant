import torch
import os
import sys
import json
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.args import ModelArgs
from parallel.model import Transformer
from llama.llama3_tok import Tokenizer
from llama.llama2_tok import Tokenizer as Tokenizer2


def start(ckpt_dir, is_llama_2, qconfig):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    torch.manual_seed(58)

    checkpoints = list(sorted(filter(lambda x: x.endswith(".pth"), os.listdir(ckpt_dir))))
    assert len(checkpoints) == model_parallel_size

    ckpt_path = os.path.join(ckpt_dir, checkpoints[get_model_parallel_rank()])
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    with open(os.path.join(ckpt_dir, "params.json")) as f:
        params = json.loads(f.read())

    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.model")
    tokenizer_cls = Tokenizer2 if is_llama_2 else Tokenizer
    tokenizer = tokenizer_cls(tokenizer_path)

    model_args = ModelArgs(**params)
    torch.set_default_dtype(torch.bfloat16)

    if is_llama_2:
        state_dict.pop("rope.freqs")
        model_args.vocab_size = tokenizer.n_words

    model = Transformer(model_args, qconfig).to("cuda")
    model.load_state_dict(state_dict)

    return model, tokenizer
