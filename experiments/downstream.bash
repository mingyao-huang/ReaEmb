#!/bin/bash
set -e

cd "$(dirname "$0")/../downstream"

gpu_id=0
dataset="games"
seed_list=(42)
llm_emb_file="../grpo_tuning/grpo_emb/${dataset}/handled/item_embs_grpo_512.pkl"
tau=2
alpha=0.01
ts_user=10
ts_item=20
emb_argu=True

model_name="sasrec_seq"

for seed in ${seed_list[@]}
do
    python main.py --dataset ${dataset} \
                --data_dir ../data \
                --model_name ${model_name} \
                --emb_argu ${emb_argu} \
                --hidden_size 128 \
                --train_batch_size 1024 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --llm_emb_file ${llm_emb_file} \
                --alpha ${alpha} \
                --tau ${tau} \
                --freeze_emb \
                --log
done
