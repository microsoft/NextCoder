from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

checkpoints = [] # add the paths to the checkpoints here


for lora_checkpoint in checkpoints[1:]:
  peft_model = AutoPeftModelForCausalLM.from_pretrained(lora_checkpoint)
  tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint)

  merged_model = peft_model.merge_and_unload()
  print(type(merged_model))
  output_path = lora_checkpoint + "-merged"
  merged_model.save_pretrained(output_path)
  tokenizer.save_pretrained(output_path)
  print(f"Model saved at {output_path}")
