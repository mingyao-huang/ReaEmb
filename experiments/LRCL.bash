#!/bin/bash
set -e

cd "$(dirname "$0")/.."

dataset="games"
lora_rank=8
lora_trainable="q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="./qwen2.5_0.5b"
your_data_path="data/${dataset}/handled/"
your_checkpoint_path="saved/${dataset}"
MAX_STEPS=2000
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
date="0616_w_R3_avg"
MAX_SOURCE_LENGTH=1024

peft_path=""

export TOKENIZERS_PARALLELISM=false
export TORCH_NAN_INF_CHECK=1

# Training Command
deepspeed --num_gpus=4 --master_port $MASTER_PORT main_llm.py \
    --deepspeed llm/ds.config \
    --do_train \
    --train_file $your_data_path/item_info.jsonline \
    --cache_dir $your_data_path \
    --prompt_column input \
    --response_column target \
    --overwrite_cache \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpoint_path/lora-$date \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps ${MAX_STEPS} \
    --logging_steps 20 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --pool_type avg \
    --dropout_ratio 0.4 \
    --user_prefix "Please provide the item description. The item information is: " \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 2 \
    --R3_think True \
    --bf16

# Testing Command
deepspeed --num_gpus=4 --master_port $MASTER_PORT main_llm.py \
    --do_predict \
    --test_file $your_data_path/item_info.jsonline \
    --peft_path $your_checkpoint_path/lora-$date/checkpoint-$MAX_STEPS \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir results/${dataset}/qwen2.5 \
    --output_file $date.json \
    --overwrite_output_dir \
    --max_source_length $MAX_SOURCE_LENGTH \
    --max_target_length 196 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --user_prefix "Please provide the item description. The item information is: " \
    --pool_type avg \
    --R3_think True
