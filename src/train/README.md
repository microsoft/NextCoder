# Model Training scripts

## Folder Structure
- `ds_config.json` contains the deepspeed configuration
- `general_acc.yaml` contains the accelerate configuration (might need to be modified as per desired system)
- `lora.py` contains the code for training model with LoRA
- `merge_lora.py` contains the code for merging trained LoRA adapters back to model for inference
- `seletkt.py` contains the code for training model with our algorithm explained in our paper
- `sft.py` contains the code for training model with Full Supervised Finetuning
  
## Usgae
### Training with SFT
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `3` and `overlap_comm` to `false` in `ds_config` for better memory optimizations
- Run the following command to start training
  ```bash
  deepspeed sft.py \
      --model_name_or_path "path to pretrained LLM" \
      --train_data_path "path to training data" \
      --output_dir "path to output dir" \
      --num_train_epochs 3 \
      --model_max_length 8192 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --save_strategy "epoch" \
      --save_steps 760 \
      --save_total_limit 25 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --logging_steps 5 \
      --report_to "wandb" \
      --gradient_checkpointing True \
      --deepspeed ds_config.json \
      --bf16 True \
      --run_name "Run name for logs" \
      --debug True \
  ```
  Update the above command as per the model
- To train on conversation data by only applying loss on the response, uncomment the lines 175, 176 and 185 and run the same command with proper dataset path

### Training with LoRA
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `2` and `overlap_comm` to `false` in `ds_config` for better memory optimizations
- Run the following command to start training
  ```bash
  deepspeed lora.py \
      --model_name_or_path "path to pretrained LLM" \
      --train_data_path "path to training data" \
      --output_dir "path to output dir" \
      --num_train_epochs 3 \
      --model_max_length 8192 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --save_strategy "epoch" \
      --save_steps 760 \
      --save_total_limit 25 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --logging_steps 5 \
      --report_to "wandb" \
      --gradient_checkpointing True \
      --deepspeed ds_config.json \
      --bf16 True \
      --run_name "Run name for logs" \
      --debug True \
  ```
  Update the above command as per the model
- Put the path of output LoRA adapters inside `merge_lora.py` and run following to get the final checkpoints
  ```bash
  python merge_lora.py
  ```

### Training with SeleKT
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `2` and `overlap_comm` to `false` in `ds_config` for better memory optimizations
- Run the following command to start training
  ```bash
  accelerate launch \
      --config_file=general_acc.yaml \
      selekt.py \
      --model_name_or_path "path to pretrained LLM" \
      --base_model_path "path to pretrained LLM" \
      --train_data_path "path to training data" \
      --output_dir "path to output directory" \
      --num_train_epochs 3 \
      --model_max_length 8192 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --save_strategy "steps" \
      --save_steps "Enter the periodicity value M for seleKT"  \
      --save_total_limit 50 \
      --learning_rate 1e-5 \
      --warmup_ratio 0.1 \
      --logging_steps 5 \
      --report_to "wandb" \
      --gradient_checkpointing True \
      --deepspeed ds_config.json \
      --bf16 True \
      --run_name "Name for logs" \
      --debug True \
      --alpha "Enter value for desired alpha parameter for SeleKT" \
  ```
  Update the above command as per the model
- To train on conversation data by only applying loss on the response, uncomment the lines 291, 292 and 301 and run the same command with proper dataset path