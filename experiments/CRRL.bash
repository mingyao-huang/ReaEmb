#!/bin/bash
set -e

cd "$(dirname "$0")/../grpo_tuning"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset="games"
# Run training with default parameters
torchrun --nproc_per_node=4 --master_port=12345 train_noise_grpo.py \
    --base_model "../qwen2.5_0.5b" \
    --lora_path "../saved/${dataset}/lora-0616_w_R3_avg/checkpoint-2000" \
    --output_dir "./output_grpo/${dataset}/" \
    --dataset ${dataset} \
    --r3_think True \
    --pool_type avg \
    --cutoff_len 128 \
    --batch_size 32 \
    --micro_batch_size 4 \
    --num_generations 8 \
    --candidate_num 10 \
    --pos_num 1 \
    --num_iterations 2 \
    --num_epochs 1 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --beta 0.00 \
    --lr 1e-4 \
    --seed 42 \
    --save_steps 1000 \
    --save_total_limit 2
    #--resume_from_checkpoint "./output_grpo/${dataset}/checkpoint-8400" \

python eval_noise_grpo.py \
    --dataset ${dataset} \
    --model_path "./output_grpo/${dataset}/" \
    --base_model "../qwen2.5_0.5b" \
    --r3_think True \
    --pool_type avg \
    --cutoff_len 128 \
    --batch_size 4
