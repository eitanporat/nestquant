MASTER_ADDR=localhost MASTER_PORT=12355 RANK=0 WORLD_SIZE=1 python -m parallel.hessian \
   --ckpt-path=/root/.llama/checkpoints/Llama3.1-8B/original  \
   --is-llama-2 False  \
   --store-path hessian_llama_3_1_8B \
   --chunk-size 10000