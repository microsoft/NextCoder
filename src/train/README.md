# Model Training scripts

## Folder Structure
- `configs` contains the deepspeed and accelerate configurations (modifialbe as per the system)
- `lora` contains the code for training model with LoRA
- `seletkt` contains the code for training model with SeleKT algorithm explained in our paper
- `sft` contains the code for training model with Full Supervised Finetuning
  
## Usgae
### Preparing the dataset
- Download the both Instruction and Conversational variant dataset from huggingface
  - [microsoft/NextCoderDataset](https://huggingface.co/datasets/microsoft/NextCoderDataset)
  - [microsoft/NextCoderDataset-Conversational](https://huggingface.co/datasets/microsoft/NextCoderDataset-Conversational)

- Run the `data_prep.py` script with the corresponding data path for instruction-variant, this will fetch the commitpackft subset and create a complete dataset for instruction-tuning
  ```bash
  python data_prep.py --commitpackft_mapping ../data/commitpackft_subset.csv --save_dir .
  ```
  This will save the final instruction dataset as `instruction_dataset` which will be used in trainig for stage-1

### Training with SFT
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `3` and `overlap_comm` to `false` in `ds_config` for better memory optimizations
- Add the respecitive variables like `MODEL_PATH`, `TRAIN_DATA`, `OUTPUT_DIR` etc. in the `run.sh` script and run
```bash
bash ./sft/run.sh
```

### Training with LoRA
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `2` and `overlap_comm` to `false` in `ds_config`
- Add the respecitive variables like `MODEL_PATH`, `TRAIN_DATA`, `OUTPUT_DIR` etc. in the `run.sh` script and run
```bash
bash ./lora/run.sh
```
>`lora/lora.py` uses `use_reentrant: True` for gradient checkpointing, and this can allow using deepspeed zero-3 optimization for large models.

### Training with SeleKT
- modify or replace the `general_acc.yaml` file as per the desired system configuration
- set the `zero_optimization-stage` to `3` and `overlap_comm` to `false` in `ds_config` for better memory optimizations
- Add the respecitive variables like `MODEL_PATH`, `TRAIN_DATA`, `OUTPUT_DIR` etc. in the `run.sh` script and run
```bash
bash ./selekt/run.sh