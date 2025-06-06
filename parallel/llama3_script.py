import torch
import os
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to model directory")
    parser.add_argument("--target-dir", type=str, required=True,
                        help="Parth to the directory with updated model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ckpt_dir = args.ckpt_path
    target_dir = args.target_dir

    os.makedirs(target_dir, exist_ok=True)
    checkpoints = list(sorted(filter(lambda x: x.endswith(".pth"), os.listdir(ckpt_dir))))
    emb = []
    for filename in tqdm(checkpoints):
        state_dict = torch.load(os.path.join(ckpt_dir, filename), weights_only=True)
        emb.append(state_dict["tok_embeddings.weight"])
    emb = torch.cat(emb, dim=0)
    emb = torch.chunk(emb, len(checkpoints), dim=1)
    for i, filename in enumerate(tqdm(checkpoints)):
        state_dict = torch.load(os.path.join(ckpt_dir, filename), weights_only=True)
        state_dict["tok_embeddings.weight"] = emb[i]
        torch.save(state_dict, os.path.join(target_dir, filename))


if __name__ == "__main__":
    main()
