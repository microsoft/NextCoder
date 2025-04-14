import os
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from transformers.trainer_callback import TrainerControl, TrainerState
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
    TrainerCallback,
    PreTrainedTokenizer
)
from transformers.trainer_utils import get_last_checkpoint
import argparse
from typing import Optional
from deepspeed.accelerator import get_accelerator
from liger_kernel.transformers import AutoLigerKernelForCausalLM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using SFT")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--train_data_path", type=str, required=True,
                      help="Path to training data file")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save the model")
    parser.add_argument("--cache_dir", type=str, required=False,
                      help="Directory for caching")
    parser.add_argument("--num_train_epochs", type=int, default=8,
                      help="Number of training epochs")
    parser.add_argument("--model_max_length", type=int, default=8192,
                      help="Maximum sequence length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                      help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--save_strategy", type=str, default="steps",
                      help="The checkpoint save strategy to use")
    parser.add_argument("--save_steps", type=int, default=500,
                      help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=25,
                      help="Limit the total amount of checkpoints")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Initial learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                      help="Linear warmup over warmup_ratio fraction of total steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                      help="Log every X updates steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                      help="The scheduler type to use")
    parser.add_argument("--report_to", type=str, default="wandb",
                      help="The integration to report the results and logs to")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                      help="Whether to use gradient checkpointing to save memory")
    parser.add_argument("--deepspeed", type=str, default=None,
                      help="Path to deepspeed config file")
    parser.add_argument("--bf16", type=bool, default=True,
                      help="Whether to use bf16 mixed precision training")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--use_liger", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--packing", type=bool, default=True,
                      help="Whether to use packing for training")
    
    args, _ = parser.parse_known_args()
    return args

def load_model_and_tokenizer(args):
    """Load the model and tokenizer with proper configuration"""
    if args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(args.model_name_or_path,
                                                          use_cache=False,
                                                          torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",)
    else:   
      model = AutoModelForCausalLM.from_pretrained(
          args.model_name_or_path,
          use_cache=False,
          torch_dtype=torch.bfloat16,
          attn_implementation="flash_attention_2",
      )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        use_fast=True,)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id


    model.is_parallelizable = True
    model.model_parallel = True
    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def setup_training_data(args, local_rank: int, tokenizer) -> Optional[torch.utils.data.Dataset]:
    """Setup and preprocess the training data"""
    print(f"Loading dataset from {args.train_data_path}")
    dataset = load_from_disk(args.train_data_path)
    return dataset

class Callback(TrainerCallback):
    def __init__(self, flush_steps=None):
        self.flush_steps = flush_steps

    def on_step_end(self, args, state, control, model, processing_class , **kwargs):
        # import sys; sys.exit(0)
        if state.global_step % self.flush_steps == 0:
            get_accelerator().empty_cache()
            if dist.is_initialized():
                dist.barrier()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Training logs at step {state.global_step}: {logs}")
      
    def on_save(self, args, state, control, **kwargs):
        print(f"Saving model at step {state.global_step}")
        

def main():
    args = parse_args()
    print("Starting training script")
    print(f"Parsed arguments: {vars(args)}")

    # Setup training configuration
    print("Setting up training configuration")

    # Setup training configuration
    training_config = SFTConfig(
        dataset_text_field="messages",
        max_seq_length=args.model_max_length,
        packing=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        output_dir=args.output_dir,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        dataset_num_proc=80,
        run_name=args.run_name,
        use_liger=args.use_liger,
        # include_num_input_tokens_seen=True, # keep it False, raised a PR to hugingface that fixes it
        )
    # Setup distributed training environment
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load dataset
    dataset = setup_training_data(args, local_rank, tokenizer)


    # Resume from checkpoint if exists
    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f'Resuming from checkpoint: {last_checkpoint}')

    # response_template = "#RESPONSE\n"
    # collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_config,
        callbacks=[Callback(flush_steps=1)],
        # data_collator=collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    print("Training completed, saving final model")
    
    # Save final model
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        print("Model saved successfully")
    

if __name__ == "__main__":
  main()