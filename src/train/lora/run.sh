#!/bin/bash

export MODEL_NAME=""
export DESC=""

# Stage 1: Instruction Training
OUTPUT_DIR_STAGE1="./output/stage1_instruction_lora"
TRAIN_DATA_STAGE1=""
MODEL_PATH=""

# Stage 2: Conversational Training  
OUTPUT_DIR_STAGE2="./output/stage2_conversational_lora"
TRAIN_DATA_STAGE2=""

# Merged model directory
MERGED_MODEL_DIR="./output/stage1_merged"

find_latest_checkpoint() {
    local output_dir=$1
    local latest_checkpoint=$(find "$output_dir" -name "checkpoint-*" -type d | sort -V | tail -1)
    echo "$latest_checkpoint"
}

merge_lora_weights() {
    local lora_checkpoint=$1
    local output_dir=$2
    
    echo "Merging LoRA weights..."
    echo "LoRA checkpoint: $lora_checkpoint" 
    echo "Output: $output_dir"
    
    python3 merge_lora.py \
        --lora_checkpoint "$lora_checkpoint" \
        --output_dir "$output_dir" \
        --safe_serialization
    
    return $?
}

echo "Starting Stage 1: Instruction Training (LoRA)..."
echo "Model: $MODEL_PATH"
echo "Training data: $TRAIN_DATA_STAGE1"
echo "Output directory: $OUTPUT_DIR_STAGE1"

mkdir -p $OUTPUT_DIR_STAGE1

# Stage 1: LoRA Instruction Training
accelerate launch \
      --config_file=../configs/general_acc.yaml \
      lora.py \
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
      --run_name "${MODEL_NAME}_stage1_instruction_lora" \

if [ $? -ne 0 ]; then
    echo "Error: Stage 1 training failed!"
    exit 1
fi

echo "Stage 1 completed successfully!"

# Find latest checkpoint
LATEST_CHECKPOINT=$(find_latest_checkpoint "$OUTPUT_DIR_STAGE1")

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $OUTPUT_DIR_STAGE1"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"

# Merge LoRA weights with base model
mkdir -p $MERGED_MODEL_DIR
merge_lora_weights "$LATEST_CHECKPOINT" "$MERGED_MODEL_DIR"

if [ $? -ne 0 ]; then
    echo "Error: LoRA merging failed!"
    exit 1
fi

echo "LoRA weights merged successfully!"
echo "Starting Stage 2: Conversational Training (LoRA)..."
echo "Model: $MERGED_MODEL_DIR"
echo "Training data: $TRAIN_DATA_STAGE2"
echo "Output directory: $OUTPUT_DIR_STAGE2"

mkdir -p $OUTPUT_DIR_STAGE2

# Stage 2: LoRA Conversational Training
accelerate launch \
      --config_file=../configs/general_acc.yaml \
      lora.py \
      --model_name_or_path "${MERGED_MODEL_DIR}" \
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
      --run_name "${MODEL_NAME}_stage2_conversational_lora" \
      --is_conversational_training \

if [ $? -ne 0 ]; then
    echo "Error: Stage 2 training failed!"
    exit 1
fi

echo "Stage 2 training completed successfully!"

# Find final checkpoint and merge again
FINAL_CHECKPOINT=$(find_latest_checkpoint "$OUTPUT_DIR_STAGE2")
FINAL_MERGED_DIR="./output/final_merged_model"

if [ ! -z "$FINAL_CHECKPOINT" ]; then
    echo "Merging final LoRA weights..."
    mkdir -p $FINAL_MERGED_DIR
    merge_lora_weights "$FINAL_CHECKPOINT" "$FINAL_MERGED_DIR"
    echo "Final merged model saved in: $FINAL_MERGED_DIR"
else
    echo "Warning: No final checkpoint found, using stage 2 output directory"
fi

echo "Both training stages completed successfully!"
echo "LoRA adapters saved in: $OUTPUT_DIR_STAGE2"
echo "Final merged model saved in: $FINAL_MERGED_DIR"