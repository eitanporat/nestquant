PYTHONPATH=/workspace/nestquant MASTER_ADDR=localhost MASTER_PORT=12355 RANK=0 WORLD_SIZE=1 python parallel/sequential_quantize.py \
    --ckpt-path /root/.llama/checkpoints/Llama3.1-8B/original \
    --hess-path /workspace/nestquant/hessian_llama_3_1_8B \
    --is-llama-2 False \
    --store-path quantized_model --seqlen 2048 \
    --quant-act \
    --quant-kv \
    --q 14 \
    --use-scalar \
    --use-j
#    --use-scalar