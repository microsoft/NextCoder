#!/usr/bin/env python3

import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--lora_checkpoint", type=str, required=True,
                       help="Path to the LoRA checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the merged model")
    parser.add_argument("--max_shard_size", type=str, default="5GB",
                       help="Maximum size of each shard when saving")
    parser.add_argument("--safe_serialization", action="store_true", default=True,
                       help="Use safe serialization format")
    return parser.parse_args()

def merge_lora_weights(lora_checkpoint, output_dir, max_shard_size="5GB", safe_serialization=True):
    """
    Merge LoRA adapter weights with the base model
    """
    print(f"Loading LoRA model from: {lora_checkpoint}")
    
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        lora_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading tokenizer from: {lora_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint)
    
    print("Merging LoRA adapters with base model...")
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    merged_model.save_pretrained(
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Successfully merged and saved model to: {output_dir}")
    
    del peft_model, merged_model
    torch.cuda.empty_cache()
    
    return output_dir

def main():
    args = parse_args()
    
    try:
        merge_lora_weights(
            lora_checkpoint=args.lora_checkpoint,
            output_dir=args.output_dir,
            max_shard_size=args.max_shard_size,
            safe_serialization=args.safe_serialization
        )
    except Exception as e:
        print(f"❌ Error during merging: {str(e)}")
        raise e

if __name__ == "__main__":
    main()