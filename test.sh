#!/bin/bash

model_name=pdf_model.iter510.bin

python3 -u exp.py \
    --mode test \
    --load_model saved_models/pdf/${model_name} \
    --beam_size 100 \
    --save_decode_to decodes/pdf/${model_name}.test.decode \
    --decode_max_time_step 110
