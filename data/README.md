# Data Generation

## Folder Structure
- `propmpts` contain all the different prompts used in synthetic data generation
- `config` contains the yaml file to map prompts to their corresponding location
- `utils.py` contains the helper code to extract and parse data from LLM responses
- `data_pipeline.py` contains the main source code for generating synthetic data according to the pipeline explained in our paper.

# Usage
- Make sure the proper packages are installed via the `environment.yaml` file provided at root folder
- Run the following command for generating data with Llama-3.3-70B-Instruct Model
  ```bash
  python data_pipeline.py --output_dir /path_to_output --language "python" --data_path /path_to_seed_code --llm_path huggingface/local path to LLM
  ```