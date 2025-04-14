import os
import json
import argparse
import uuid
import jsonlines
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import logging
import colorlog
from openai import AzureOpenAI, BadRequestError, APIStatusError, OpenAIError
import re
import time
from azure.identity import AzureCliCredential, get_bearer_token_provider, ChainedTokenCredential, DefaultAzureCredential
from utils import PromptLoader, CodeExtractor, generate_diff, get_start_idx

DEBUG = False
REQUIRED_SAMPLES=50000

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
mode = "a"
if DEBUG:
    mode = "w"

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(handler)    

# Environment setup (comment them for servers)
os.environ['HF_HOME'] = '/mnt/datadrive/guestuser/t-swsingh/nofuncoder/cache'
os.environ['OUTLINES_CACHE_DIR'] = '/mnt/datadrive/guestuser/t-swsingh/miniconda3/.cache/outlines'

# Constants
LANGUAGES = ['python', 'java', 'javascript', 'go', 'rust']
MAX_ATTEMPTS = 2

# Initialize utilities
prompt_loader = PromptLoader("config/prompts.yml")
output_parser = CodeExtractor()

def load_seed_dataset(language, num_samples, split, path='', start_idx=0, reverse=False):
    if path:
        logger.info(f"Loading dataset from path: {path} for language: {language}")
        ds = load_from_disk(path).filter(lambda x: x['language'] == language.strip(), num_proc=80)
        if reverse:
            return ds.select(range(len(ds) - start_idx - 1, -1, -1))
        else:
          return ds.select(range(start_idx, len(ds)))


def initialize_openai():
    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(ChainedTokenCredential(
    AzureCliCredential(),
    DefaultAzureCredential(
        exclude_cli_credential=True,
        exclude_environment_credential=True,
        exclude_shared_token_cache_credential=True,
        exclude_developer_cli_credential=True,
        exclude_powershell_credential=True,
        exclude_interactive_browser_credential=True,
        exclude_visual_studio_code_credentials=True,
        managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
    )
    ),scope)
    api_version = '2024-10-21'
    model_name = 'gpt-4o'
    model_version = '2024-05-13'
    deployment_name = re.sub(r'[^a-zA-Z0-9-_]', '', f'{model_name}_{model_version}')
    endpoint = f'' # pin the appropriate endpoint

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,
        api_version=api_version,
    )
    
    return client

def initialize_llm(model_path, decoding_strategy, max_new_tokens):
    llm = LLM(model_path, tensor_parallel_size=4, swap_space=4, gpu_memory_utilization=0.95)
    tokenizer = llm.get_tokenizer()
    max_new_tokens = tokenizer.model_max_length
    
    if decoding_strategy == "greedy":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    else:
        sampling_params = SamplingParams(temperature=0.6, max_tokens=max_new_tokens, top_k=50, top_p=0.9, ignore_eos=False)
    sampling_params.stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    return llm, sampling_params


def open_ai_generate(llm, prompt):
    try:
        logger.debug(f"Prompt: {prompt}")
        gpt_4o_deployment = "gpt-4o_2024-05-13"
        messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
        response = llm.chat.completions.create(
            model=gpt_4o_deployment,
            messages=messages,
            max_tokens = args.max_new_tokens,
            top_p = 0.9,
            temperature = 0.6,
            seed=42,
        )
        generated_text = response.choices[0].message.content
        return generated_text
    except Exception as e:
        logger.warning(f"OpenAI generation failed: {e}")
        raise

def generate_with_retry(llm, sampling_params, prompt_key, parser_func, args=None, **kwargs) -> dict:
    base_prompt = prompt_loader.format_prompt(prompt_key, **kwargs)
    retry_count = 1
    while True:
      try:
          prompt = base_prompt
          if args.openai_api == "true":
              generated_text = open_ai_generate(llm, prompt)
          else:
              logger.debug(sampling_params)
              tokenizer = llm.get_tokenizer()
              conversation = tokenizer.apply_chat_template(
                  [{'role': 'user', 'content': prompt}],
                  tokenize=False)
              if len(tokenizer.encode(conversation)) > tokenizer.model_max_length:
                  logger.warning(f"Length of prompt exceeds the model's window, skipping....")
                  return None
              outputs = llm.generate([conversation], sampling_params)
              generated_text = outputs[0].outputs[0].text
              logger.warning(f"length of prompt token: {len(outputs[0].prompt_token_ids)}")
          return parser_func(generated_text)
      except OpenAIError  as e:
          error_str = str(e)
          if 'error code: 429' in error_str.lower() and 'nocapacity' in error_str.lower():
              retry_count += 1
              logger.warning(f"NoCapacity error encountered. Attempt {retry_count}. Error: {e}")
              logger.warning("Sleeping for 6 mins before retry...")
              time.sleep(360)
              continue
          else:
              logger.warning(f"OpenAI error occurred (not NoCapacity). Error: {e}")
              return None
      except Exception as e:
          logger.warning(f"Non-OpenAI error occurred for {prompt_key}. Error: {e}")
          logger.debug(f"Raw LLM output:\n{generated_text}")
          return None

def generate_problem_code_pair(llm, sampling_params, lang, seed_code, level, selected_areas, args):
    prompt_key = "problem_code"
    kwargs = {"lang": lang, 
              "seed_code": seed_code, 
              "level": level, 
              "area1_name":selected_areas[0][0],
              "area1_desc":selected_areas[0][1],
              "area2_name":selected_areas[1][0],
              "area2_desc":selected_areas[1][1],
              "area3_name":selected_areas[2][0],
              "area3_desc":selected_areas[2][1],
              }
    result = generate_with_retry(llm, sampling_params, prompt_key, output_parser.problem_code_extractor, args, **kwargs)
    if result:
        logger.info("Generated problem-code pair:")
        logger.info(f"Problem: {result['problem_statement']}")
        logger.info(f"Code: {result['original_code']}")
        logger.info(f"Metadata: {result['metadata']}")
        return result
    return None

def generate_code_edit(llm, sampling_params, problem, code, lang, metadata, selected_areas, args):
    prompt_key = "code_edit_generation"
    kwargs = {"problem": problem, "code": code, "lang": lang, "metadata": metadata,
              "area1_name":selected_areas[0][0],
              "area1_desc":selected_areas[0][1],
              "area2_name":selected_areas[1][0],
              "area2_desc":selected_areas[1][1],
              "area3_name":selected_areas[2][0],
              "area3_desc":selected_areas[2][1],
              }
    result = generate_with_retry(llm, sampling_params, prompt_key, output_parser.code_edit_extractor, args, **kwargs)
    if result:
        logger.info("Generated code edit:")
        for k in result:
            logger.info(f"{k}: {result[k]}")
        return result
    return None

def generate_instruction(llm, sampling_params, problem, lang, code, edited_code, explanations, args):
    prompt_key = "generate_instructions"
    kwargs = {"problem": problem, "lang": lang, "code": code, "edited_code": edited_code, "explanations": explanations}
    result = generate_with_retry(llm, sampling_params, prompt_key, output_parser.instructions_extractor, args, **kwargs)
    if result:
        logger.info("Generated instruction:")
        for k in result:
            logger.info(f"{k}: {result[k]}")
    return result if result else None


def quality_check(llm, sampling_params, lang, seed_code, problem, code, p_edit1, detailed_instruction, human_instruction, conversational_instruction, args):
    prompt_key = "quality_check"
    kwargs = {"lang": lang, "seed_code": seed_code, "problem": problem, "original_code": code, "preferred_edit": p_edit1, "detailed_instruction": detailed_instruction, 
              "human_instruction": human_instruction, "conversational_instruction": conversational_instruction}
    result = generate_with_retry(llm, sampling_params, prompt_key, output_parser.quality_check_extractor, args, **kwargs)
    if result:
        logger.info("Quality check results: ")
        for k in result:
            logger.info(f"{k}: {result[k]}")
        return result
    return None

def calculate_final_result(extracted_quality):
    try:
      for i in extracted_quality:
        print(extracted_quality[i])
      scores = [float(score['score']) if score['score'] else 0 for score in extracted_quality.values()]
      avg_score = sum(scores) / len(scores)
      passed = avg_score >= 7 and all(score >= 5 for score in scores)
      
      return {
          'average_score': round(avg_score, 2),
          'passed': passed,
          'individual_scores': scores
      }
    except Exception as e:
      logger.warning(f"Failed to calculate final result. Error: {e}")
      return None

def process_item(llm, sampling_params, item, lang, args):
    SEED = 42
    seed_code = item['content']
    levels = ['function', 'class', 'file-level']
    improvement_areas = {
    "Bug Fixes": """Code should have multiple layers of significant issues including:
        - Completely unimplemented critical functions
        - Partially implemented functions with incorrect logic
        - Missing core features described in problem statement
        - Concurrency and race condition vulnerabilities
        - Data consistency and transaction atomicity issues
        - Architectural flaws (missing important components/systems)
        - Critical validation functions either missing or always returning True/False
        - Float precision issues with numerical operations
        - Functions silently failing or returning incorrect values
        The code should be structured enough to understand the intent but have clear fundamental flaws that require significant fixes.""",
    
    "Performance": "Code has inefficient algorithms, unnecessary loops, redundant computations, or poor data structure choices that impact execution speed",
    
    "Resource Management": "Inefficient memory usage, resource leaks, unclosed connections/files, excessive memory allocations, or missing cleanup operations",
    
    "Runtime Optimization": "Poor time complexity, unnecessary API calls, unoptimized database queries, or inefficient string/array operations",
    
    "Maintainability": "Poor code organization, lack of modularity, inconsistent naming conventions, missing documentation, or code duplication",
    
    "Security Enhancements": "Potential security vulnerabilities, unsanitized inputs, unsafe data handling, missing access controls, or improper authentication",
    
    "General Improvements": "Code readability issues, hardcoded values, lack of configuration options, missing logging/monitoring capabilities"
    }

    logger.info(f"Processing item for language: {lang}")

    level = random.choice(levels)
    bug_fixes = ("Bug Fixes", improvement_areas["Bug Fixes"])
    remaining_areas = {k: v for k, v in improvement_areas.items() if k != "Bug Fixes"}
    other_areas = random.sample(list(remaining_areas.items()), 2)    
    selected_areas = [bug_fixes] + other_areas

    # Phase-1: Generating Problem-Code Pair
    result = generate_problem_code_pair(llm, sampling_params, lang, seed_code, level, selected_areas, args)
    if result:
      problem = result['problem_statement']
      code = result['original_code']
      metadata = result['metadata']
    else:
      logger.warning("Failed to generate problem-code pair.")
      return None

    # Phase-2: Generating Code Edit
    result = generate_code_edit(llm, sampling_params, problem, code, lang, metadata, selected_areas, args)
    if result:
        p_edit1 = result['preferred_edit_a']
        p_edit2 = result['preferred_edit_b']
        np_edit = result['non_preferred_edit']
        explanation = result['explanation']
    else:
        logger.warning("Failed to generate code edit.")
        return None
    
    # Phase-3: Generating Instructions
    edit_1_explanation = explanation.split("PREFERRED_1:")[-1].split("PREFERRED_2:")[0].strip()

    result = generate_instruction(llm, sampling_params, problem, lang, code, p_edit1, edit_1_explanation, args)
    if result:
        detailed_instruction = result['detailed_instruction']
        concise_instruction = result['concise_instruction']
        human_instruction = result['human_instruction']
        conversational_instruction = result['conversational_instruction']
    else:
        logger.warning("Failed to Generate Instructions.")
        return None

    # Phase-4: Quality Check
    result = quality_check(llm, sampling_params, lang, seed_code, problem, code, p_edit1, detailed_instruction, human_instruction, conversational_instruction, args)

    if result:
        extracted_quality = result['scores']
        final_result = calculate_final_result(extracted_quality)
        if final_result:
            if final_result['passed'] and final_result['average_score'] > 7:
                logger.info("Successfully calculated final result.")
            else:
                logger.warning("Sample is not of good quality.")
                logger.info("Final Result:")
                for k in final_result:
                    logger.info(f"{k}: {final_result[k]}")
        else:
            logger.warning("Failed to calculate final result.")
            return None

    

    logger.info("Successfully processed item.")
    return {
        "_id": uuid.uuid4().hex,
        "seed_code_id": item['id'],
        "language": lang,
        "problem": problem,
        "original_code": code,
        "preferred_edit_a": p_edit1,
        "preferred_edit_b": p_edit2,
        "non_preferred_edit": np_edit,
        "preferred_diff_a": generate_diff(code, p_edit1),
        "preferred_diff_b": generate_diff(code, p_edit2),
        "non_preferred_diff": generate_diff(code, np_edit),
        "improvement_areas": improvement_areas,
        "seed_code": seed_code,
        "instructions": [detailed_instruction, concise_instruction, human_instruction],
        "conversational_instruction": conversational_instruction,
        "edit_explanation": explanation,
    }

def main(args):
    if args.openai_api == "true":
        llm = initialize_openai()
        sampling_params = None
        args.max_new_tokens = 4096
        reverse_loading = False
    else:
        llm, sampling_params = initialize_llm(args.llm_path, args.decoding_strategy, 8192)
        reverse_loading = True
    if args.language is not None:
        LANGUAGES = [args.language]
    for language in LANGUAGES:
        output_file = os.path.join(args.output_dir, f"{language}.jsonl")
        
        start_idx = 0
        if not DEBUG and os.path.exists(output_file):
            start_idx = get_start_idx(output_file, reverse_loading)
        logger.info(f"Attempting to generate {REQUIRED_SAMPLES} samples for {language}")
        logger.info(f"Starting from index: {start_idx}")
        dataset = load_seed_dataset(language, args.num_samples + 100, split="train", path=args.data_path, start_idx=start_idx, reverse=reverse_loading)

        
        f_mode = "a"
        if DEBUG:
            f_mode = "w"
        logger.info(f"{'Writing' if (DEBUG and start_idx == 0) else 'Appending'} output to: {output_file} at index: {start_idx} with mode {f_mode}")
        with open(output_file, mode=f_mode) as f:
            f.seek(0, 2)
            f.write("\n")
        with jsonlines.open(output_file, mode=f_mode) as writer:
            logger.info(f"Running total seed_codes of {len(dataset)}")
            for idx, item in enumerate(tqdm(dataset, desc=f"Processing {language}"), start=1):
                result = process_item(llm, sampling_params, item, language, args)
                if result:
                    writer.write(result)
                    logger.info(f"Successfully added item {idx} for {language}.")
                else:
                    logger.warning(f"Failed to process item {idx} for {language}.")
                # import sys; sys.exit(0)
        logger.info(f"Completed processing for language: {language}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='dataset',
                        help="path to output dir for dumping jsons")
    parser.add_argument('--llm_path', type=str, default='',
                        help="path to LLM model")
    parser.add_argument('--data_path', type=str, default='', required=True, help="path to seed dataset")
    parser.add_argument('--num_samples', type=int, default=50000,
                        help="number of samples to process per language")
    parser.add_argument('--decoding_strategy', type=str, default="sampling",
                        choices=["greedy", "sampling"], help="decoding strategy for LLM")
    parser.add_argument('--log_level', type=str, default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument('--seed', type=int, default=0, help="seed for generation")
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens to generate")
    parser.add_argument('--openai_api', type=str, default="false", help="Whether to use OpenAI API")
    parser.add_argument('--deployment_id', type=str, default=None, help="")
    parser.add_argument('--language', type=str, default=None, help="Individual language to process")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)
    file_handler = logging.FileHandler(f"{args.output_dir}/{args.language}.log", mode=mode)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
