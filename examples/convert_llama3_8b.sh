#!/bin/bash

python -m EasyLM.models.llama.convert_hf_to_easylm \
    --hf_model='/home/zephyr/gcs-bucket/model_ckpts/Llama-3.1-8B/' \
    --output_file='/home/zephyr/gcs-bucket/model_ckpts/pruning/easylm/llama3_8b' \
    --streaming=True \
    --llama.base_model='llama3_8b'