#!/bin/bash


export MODEL_NAME=""
export DESC=""

OUTPUT_DIR=""
TRAIN_DATA=""
MODEL_PATH=""

mkdir -p $OUTPUT_DIR

accelerate launch \
      --config_file=../configs/general_acc.yaml \
      sft.py \
      --model_name_or_path "$MODEL_PATH" \
      --train_data_path "$TRAIN_DATA" \
      --output_dir ${OUTPUT_DIR} \
      --num_train_epochs 3 \
      --model_max_length 16384 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --save_strategy "epoch" \
      --save_steps 760 \
      --save_total_limit 25 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --weight_decay 0.1 \
      --logging_steps 5 \
      --lr_scheduler_type "cosine" \
      --report_to "wandb" \
      --gradient_checkpointing True \
      --deepspeed ../configs/ds_config.json \
      --bf16 True \
      --run_name "" \



accelerate launch \
      --config_file=../configs/general_acc.yaml \
      sft.py \
      --model_name_or_path "${MODEL_PATH}" \
      --train_data_path "$TRAIN_DATA" \
      --output_dir ${OUTPUT_DIR} \
      --num_train_epochs 3 \
      --model_max_length 16384 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --save_strategy "epoch" \
      --save_steps 760 \
      --save_total_limit 25 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --weight_decay 0.1 \
      --logging_steps 5 \
      --lr_scheduler_type "cosine" \
      --report_to "wandb" \
      --gradient_checkpointing True \
      --deepspeed ../configs/ds_config.json \
      --bf16 True \
      --run_name "" \
      --is_conversational_training \

