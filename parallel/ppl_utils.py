import torch
from torch import nn
import datasets
from tqdm import tqdm


def get_wikitext2(tokenizer, is_testset):
    split = "test" if is_testset else "train"
    dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[split]
    text = "\n\n".join(dataset["text"])

    encoded_ids = tokenizer.encode(
        text,
        bos=True, 
        eos=True 
    )

    input_ids = torch.tensor(encoded_ids, dtype=torch.long)

    return input_ids


def split_dataset(dataset, seqlen):
    total_len = dataset.shape[0]
    nsamples = total_len // seqlen
    dataset = dataset[: nsamples * seqlen]
    dataset = dataset.reshape((nsamples, seqlen)).to("cuda")
    return dataset


def compute_perplexity(model, dataset, seqlen):
    dataset = split_dataset(dataset, seqlen)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    nlls = []

    for i in tqdm(range(dataset.shape[0])):
        batch = dataset[i: i+1].to("cuda")
        logits = model(batch, start_pos=0).float()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ) 

        nll = loss.view(shift_labels.size(0), -1).mean(dim=1) 
        nlls.append(nll)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    return ppl.item()
