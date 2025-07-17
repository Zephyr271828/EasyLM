#! /bin/bash

# This is the example script to serve a 7B LLaMA model on a GPU machine or
# single TPU v3-8 VM. The server will be listening on port 35009.


python -m EasyLM.models.llama.llama_serve \
    --load_checkpoint="params::/home/zephyr/gcs-bucket/model_ckpts/pruning/easylm/llama3_8b" \
    --tokenizer="/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B" \
    --mesh_dim='1,-1,1' \
    --dtype='bf16' \
    --input_length=256 \
    --seq_length=512 \
    --lm_server.batch_size=1 \
    --lm_server.port=35009 \
    --lm_server.pre_compile='all'

