# NestQuant simulation
## If Already Cloned
If you cloned without submodules, run:
```bash
git submodule update --init --recursive
```

## Build CUDA Extension

```bash
cd applied-ai/kernels/cuda/inference/hadamard_transform
python setup.py install
```

### Prerequisites

To install the conda environment needed to run scripts in this repository, run:

```bash
conda env create -f quant.yaml
conda activate quant
```

### Downloading Llama

To run the quantization of a Llama model, you need three components: checkpoint, `params.json` file, and the tokenizer config. They should be in the original format and can be downloaded from Huggingface here (for Llama-3-8B):

`https://huggingface.co/meta-llama/Meta-Llama-3-8B/tree/main/original`

### Running scripts

All scripts are run with command `torchrun --nproc_per_node GPUS -m SCRIPT -- ARGS`. The number of GPUs depends on the model and is equal to the number of checkpoint files.

Most scripts contain useful documentation in help, it can be obtained by running `python -m SCRIPT_NAME --help`.

For Llama-3-70B, please run the script `parallel.llama3_script` to convert the checkpoint into expected format.

*Computing Hessians for LDLQ.* Run the script `parallel.hessian`. The script creates a folder with files, where each file corresponds to the Hessian of each layer. If running the whole script makes the GPU to run out of memory, you can specify arguments `--L` and `--R` to describe the half-interval of Hessian indices to compute.

*Running the dynamic programming for optimizing betas.* The dynamic programming script will be added in the future. Right now, you could try running the perplexity script with default beta values.

*Computing QA-LDLQ statistics.* The QA-LDLQ statistics are computed by `parallel.err` script and are saved in a YAML file.

*Quantizing the model.* The `parallel.w_quant` script takes the path to Hessians, and (optionally) path to the QA-LDLQ statistics, and quantizes the model with a given $q$ and betas. The model is saved in a specified directory.

*Evaluation.* The perplexity on wikitext2 can be evaluated with `parallel.ppl` script.

*Zero-shot benchmark evaluation.* We will add the code for zero-shot evaluations in the future.

Please let us know if you run into any issues when using this code, or would like to see any features implemented.
