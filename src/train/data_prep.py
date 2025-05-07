from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
import argparse


def process(example):
  lang = example['lang'].lower()
  message = example['message']
  instruction = f'''
  Rewrite the given {lang} program as per the following instruction.\n
  {message}
  Write the entire code and no other text in the response.
  ```{lang}\n{example['old_contents']}
  ```
  '''

  completion = f'''
  ```{lang}
  {example['new_contents']}
  ```'''
  
  return {
    'prompt': instruction,
    'completion': completion,
  }


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--commitpackft_mapping", type=str, default="../data/commitpackft_subset.csv")
  parser.add_argument("--save_dir", type=str, default="")
  args = parser.parse_args()

  langs = ["python", "javascript", "java", "go", "c", "c++", "kotlin", "rust"]
  commitpck = []
  for lang in langs:
      commitpck.append(load_dataset("bigcode/commitpackft", lang, trust_remote_code=True))

  commitpack_final = concatenate_datasets([_["train"] for _ in commitpck])

  commitpack_subset_map = pd.read_csv(args.commitpackft_mapping)
  commitpack_subset_map.columns = ["repos", "commit"]
  commitpack_final_df = commitpack_final.to_pandas()
  matched_df = commitpack_final_df.merge(commitpack_subset_map, on=['commit', 'repos'])
  filtered_dataset = Dataset.from_pandas(matched_df)

  processed_dataset = filtered_dataset.map(process, remove_columns=filtered_dataset.column_names)

  synthetic_ins = load_dataset("microsoft/NextCoderDataset")

  final_instruction_ds = concatenate_datasets([synthetic_ins['train'], processed_dataset])
  final_instruction_ds.save_to_disk(f"{args.save_path}/instruction_dataset")

  print(f"Dataset {args.save_path}/instruction_dataset saved to disk.")