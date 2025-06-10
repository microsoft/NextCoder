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
from transformers.integrations import deepspeed_load_checkpoint
from transformers.trainer_utils import get_last_checkpoint
import argparse
import shutil
from pathlib import Path
from typing import Optional
from deepspeed.accelerator import get_accelerator 
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import deepspeed
import copy
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaConfig, Qwen2Config, Qwen2ForCausalLM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model using SFT")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--base_model_path", type=str, required=True)
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
    parser.add_argument("--alpha", type=float, default=0.05,)
    
    args, _ = parser.parse_known_args()
    return args

def load_model_and_tokenizer(args):
    """Load the model and tokenizer with proper configuration"""
    if args.use_liger:
        model = AutoLigerKernelForCausalLM.from_pretrained(args.model_name_or_path,
                                                          use_cache=False,
                                                          attn_implementation="flash_attention_2",)
    else:   
      model = AutoModelForCausalLM.from_pretrained(
          args.model_name_or_path,
          use_cache=False,
          attn_implementation="flash_attention_2",
      )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        use_fast=True,)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Model and tokenizer loaded successfully")
    return model, tokenizer

def setup_training_data(args, local_rank: int, tokenizer):
    """Setup and preprocess the training data"""
    print(f"Loading dataset from {args.train_data_path}")
    dataset = load_from_disk(args.train_data_path)
    
    return dataset

def copy_llm_files(source_dir, dest_dir):
      source_path = Path(source_dir)
      dest_path = Path(dest_dir)
      
      dest_path.mkdir(parents=True, exist_ok=True)
      
      for item in source_path.glob('**/*'):
          rel_path = item.relative_to(source_path)
          destination = dest_path / rel_path
          
          if item.is_file():
              if not (str(item).endswith('.safetensors') or str(item).endswith('safetensors.index.json')):
                  destination.parent.mkdir(parents=True, exist_ok=True)
                  shutil.copy2(item, destination)
                  print(f"Copied: {rel_path}")
          elif item.is_dir():
              destination.mkdir(parents=True, exist_ok=True)

@torch.no_grad()
def selekt(base_path, save_path, alpha, rescaling, rank, trainer):
    if rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        config = Qwen2Config.from_pretrained(base_path)
        print("Loading model arch")
        base_model = Qwen2ForCausalLM(config)
        print("Loading torch state dict")
        base_sd = torch.load(f"{base_path}/pytorch_model.bin", map_location=torch.device('cpu'))
        base_model.load_state_dict(base_sd, strict=False)
        print("state dict loaded")
        base_model.eval()
        base_model._no_split_modules = None
        base_model.model_parallel = False
        base_model.is_parallelizable = False

    model1 = trainer.model_wrapped

    if dist.is_initialized():
        dist.barrier()

    for name, param in tqdm(model1.named_parameters(), total=len(list(model1.parameters())), desc="SeleKT"):
        stripped_name = ".".join(name.split(".")[1:])  # Remove DeepSpeed wrapper prefix
        with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
            if rank == 0:
                clean_name = name.replace("module.", "").replace("_orig_mod.", "")
                base_param = base_sd[clean_name]
                base_param = base_param.to(
                    device=param.device,
                    dtype=param.dtype,
                    non_blocking=True
                )
                delta = param - base_param
                # print("+"*100)
                # print(torch.sum(delta))
                # print("+"*100)
                mask = torch.zeros_like(delta)
                _, indices = torch.topk(delta.abs().view(-1), int(alpha * delta.numel()))
                mask.view(-1)[indices] = 1

                masked_delta = delta * mask
                base_model.get_parameter(stripped_name).to("cpu").data += masked_delta.to("cpu")

    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        Path(save_path).mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(
            save_path,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(save_path)
        print(f"Saved final model to {save_path}")

        del base_model
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()



class Callback(TrainerCallback):
    def __init__(self, base_model_path, flush_steps, alpha):
        self.flush_steps = flush_steps
        self.base_model_path = base_model_path
        self.trainer = None
        self.alpha = alpha
        self.base_model = None
        self.M = 1

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % self.flush_steps == 0:
            get_accelerator().empty_cache()
            if dist.is_initialized():
                dist.barrier()

    def _selekt_and_load_pretrained(self, args, state, local_rank):
        recent_saved_checkpoint = args.output_dir + f"/checkpoint-{state.global_step}"
        output_path = recent_saved_checkpoint
        selekt(self.base_model_path, output_path, self.alpha, False, local_rank, self.trainer)
        if dist.is_initialized():
          dist.barrier()
        return output_path

    def _selekt_and_load_previous(self, args, state, local_rank):
        recent_saved_checkpoint = args.output_dir + f"/checkpoint-{state.global_step}"
        if os.path.basename(recent_saved_checkpoint) + "-selekt" in os.listdir(args.output_dir):
            return recent_saved_checkpoint + "-selekt"
        if state.epoch > 1:
          prev_save = sorted([int(i.split("-")[-2]) for i in os.listdir(args.output_dir) if i.endswith("selekt")])[-1]
          previous_checkpoint = args.output_dir + f"/checkpoint-{prev_save}-selekt"
        else:
            previous_checkpoint = self.base_model_path

        output_path = args.output_dir + f"/checkpoint-{state.global_step}-selekt"
        selekt(recent_saved_checkpoint, previous_checkpoint, output_path, self.alpha, False, local_rank)
        if dist.is_initialized():
          dist.barrier()
        return output_path

    def on_save(self, args, state, control, model=None, **kwargs):
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        print("*"*100)
        print(f"Saving model at step {state.global_step} at {args.output_dir}")

        if dist.is_initialized():
            dist.barrier()

        output_path = self._selekt_and_load_pretrained(args, state, local_rank)
        print("Loading the SeleKT checkpoint")
        try:
          if self.trainer and self.trainer.deepspeed:
            self.trainer.model_wrapped.load_checkpoint(
                    output_path,
                    load_module_strict=True,
                    load_optimizer_states=True,
                    load_lr_scheduler_states=True
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"selekt model is not loaded")

        if dist.is_initialized():
            dist.barrier()
        get_accelerator().empty_cache()
      

def train(args):
    print(f"Running with alpha {args.alpha} save strategy {args.save_strategy} save steps {args.save_steps}")
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
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=args.deepspeed,
        dataset_num_proc=80,
        run_name=args.run_name,
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    model, tokenizer = load_model_and_tokenizer(args)
    dataset = setup_training_data(args, local_rank, tokenizer)

    last_checkpoint = get_last_checkpoint(args.output_dir)
    if last_checkpoint:
        print(f'Resuming from checkpoint: {last_checkpoint}')


    collator = None
    if args.is_conversational_training:
      response_template = "#RESPONSE\n"
      collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)

    callback = Callback(base_model_path=args.base_model_path, flush_steps=1, alpha=args.alpha)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_config,
        callbacks=[callback],
        data_collator=collator,
    )
    callback.set_trainer(trainer)
    print(f"Starting training for epoch {args.num_train_epochs}")
    trainer.train(
        resume_from_checkpoint=last_checkpoint,
    )

    if dist.is_initialized():
        print("Training Completed")
        dist.barrier()

        

def main():
    args = parse_args()
    print("Starting training script")
    print(f"Parsed arguments: {vars(args)}")
    train(args)

if __name__ == "__main__":
  main()