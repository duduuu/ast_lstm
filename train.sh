#!/bin/bash
set -e

seed=${1:-0}
vocab="vocab.bin"
train_file="train.bin"
dropout=0.3
hidden_size=256
action_embed_size=256
field_embed_size=32
type_embed_size=32
lr_decay=0.5
beam_size=5
ls=0.1
model_name=pdf_model

python3 -u exp.py \
    --seed ${seed} \
    --mode train \
    --batch_size 10 \
    --asdl_file asdl/lang/pdf/pdf_asdl.txt \
    --transition_system pdf \
    --train_file data/pdf/${train_file} \
    --vocab data/pdf/${vocab} \
    --primitive_token_label_smoothing ${ls} \
    --hidden_size ${hidden_size} \
    --action_embed_size ${action_embed_size} \
    --field_embed_size ${field_embed_size} \
    --type_embed_size ${type_embed_size} \
    --dropout ${dropout} \
    --max_num_trial 5 \
    --beam_size ${beam_size} \
    --decode_max_time_step 110 \
    --log_every 50 \
    --save_to saved_models/pdf/${model_name}

