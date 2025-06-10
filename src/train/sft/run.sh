#!/bin/bash

export MODEL_NAME=""
export DESC=""

# Stage 1: Instruction Training
OUTPUT_DIR_STAGE1="./output/sft_stage1_instruction"
TRAIN_DATA_STAGE1=""
MODEL_PATH=""

# Stage 2: Conversational Training
OUTPUT_DIR_STAGE2="./output/sft_stage2_conversational"
TRAIN_DATA_STAGE2=""

find_latest_checkpoint() {
    local output_dir=$1
    local latest_checkpoint=$(find "$output_dir" -name "checkpoint-*" -type d | sort -V | tail -1)
    echo "$latest_checkpoint"
}

echo "Starting Stage 1: Instruction Training..."
echo "Model: $MODEL_PATH"
echo "Training data: $TRAIN_DATA_STAGE1"
echo "Output directory: $OUTPUT_DIR_STAGE1"

mkdir -p $OUTPUT_DIR_STAGE1

# Stage 1: Instruction Training
accelerate launch \
      --config_file=../configs/general_acc.yaml \
      sft.py \
      --model_name_or_path "$MODEL_PATH" \
      --train_data_path "$TRAIN_DATA_STAGE1" \
      --output_dir ${OUTPUT_DIR_STAGE1} \
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
      --run_name "${MODEL_NAME}_stage1_instruction" \

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 training failed!"
    exit 1
fi

echo "Stage 1 completed successfully!"

LATEST_CHECKPOINT=$(find_latest_checkpoint "$OUTPUT_DIR_STAGE1")

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR_STAGE1"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"
echo "Starting Stage 2: Conversational Training..."
echo "Model: $LATEST_CHECKPOINT"
echo "Training data: $TRAIN_DATA_STAGE2"
echo "Output directory: $OUTPUT_DIR_STAGE2"

mkdir -p $OUTPUT_DIR_STAGE2

# Stage 2: Conversational Training
accelerate launch \
      --config_file=../configs/general_acc.yaml \
      sft.py \
      --model_name_or_path "${LATEST_CHECKPOINT}" \
      --train_data_path "$TRAIN_DATA_STAGE2" \
      --output_dir ${OUTPUT_DIR_STAGE2} \
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
      --run_name "${MODEL_NAME}_stage2_conversational" \
      --is_conversational_training \


# Check if stage 2 completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Stage 2 training failed!"
    exit 1
fi

echo "Stage 2 training completed!"
echo "Both training stages completed successfully!"
echo "Final model saved in: $OUTPUT_DIR_STAGE2"