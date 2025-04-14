import re
import json
import yaml
import os
import difflib
from pathlib import Path

class PromptLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

    def load_prompt(self, prompt_name):
        prompt_path = Path(self.config['prompt_directory']) / self.config['prompts'][prompt_name]
        with open(prompt_path, 'r') as prompt_file:
            return prompt_file.read()

    def format_prompt(self, prompt_name, **kwargs):
        prompt = self.load_prompt(prompt_name)
        return prompt.format(**kwargs)

def generate_diff(original_code, edited_code):
    original_lines = original_code.splitlines()
    edited_lines = edited_code.splitlines()
    
    diff = difflib.unified_diff(
        original_lines, edited_lines, 
        fromfile='original_code', tofile='edited_code', 
        lineterm=''
    )
    
    return '\n'.join(diff)


class OutputParser:
    def __init__(self):
        pass

    def detect_quote_type(self, string):
        string = string.strip()
        
        if string.startswith('"""'):
            return '"""'
        elif string.startswith("'''"):
            return "'''"
        elif string.startswith('"'):
            return '"'
        elif string.startswith("'"):
            return "'"
        else:
            return None
        
    def balance_braces(self, s: str):
        count = 0
        for i, char in enumerate(s):
            if char == '{':
                count += 1
            elif char == '}':
                count -= 1
                if count == 0:
                    return s[:i+1]
        return None
    
    def json_parser(self, text):  
        start = text.find('{')
        if start == -1:
            raise ValueError("No JSON-like structure found in the text")
        
        json_str = self.balance_braces(text[start:])
        if json_str is None:
            raise ValueError("Unbalanced braces in the JSON-like structure")

        # Remove trailing commas before closing braces or square brackets
        self.json_str = re.sub(r',\s*([}\]])', r'\1', json_str)      
        # data = json.loads(text)
        if '"Feedback":' in self.json_str:
            a = self.json_str.split('"Feedback": ')[-1].strip()
            quote = self.detect_quote_type(a)
            a = a.replace(quote, "'''")
        data = eval(self.json_str)
        return data

    def problem_code_parser(self, text):
        try:
            data = self.json_parser(text)
            if 'problem' not in data or 'code' not in data:
                raise ValueError("Parsed JSON is missing 'problem' or 'code' keys")
            return data
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempting to salvage data from malformed JSON...")
            problem = self.json_str.split('"problem":')[-1].split('"code":')[0].strip()
            start_quote = self.detect_quote_type(problem)
            if start_quote is not None:
                problem = problem.replace(start_quote, "")[:-1]
            
            code = self.json_str.split('"code":')[-1].strip()
            if code.endswith("}"):
                code = code.strip()[:-1].strip()
            
            start_quote = self.detect_quote_type(code)
            if start_quote is not None:
                code = code.replace(start_quote, "")
     
            code = code if code != "" else None
            # code = code_match.group(2) if code_match else None
            
            if problem or code:
                return {
                    "problem": problem,
                    "code": code
                }
            else:
                raise ValueError("Failed to extract problem or code from malformed JSON")
            
    def code_edit_parser(self, text):
        try:
            data = self.json_parser(text)
            if 'improvement_areas' not in data or 'edited_code' not in data:
                raise ValueError("Parsed JSON is missing 'problem' or 'code' keys")
            return data
        except Exception as e:
            print(f"JSON parsing error: {e}")
            print(f"Attempting to salvage data from malformed JSON...")
            problem = self.json_str.split('"improvement_areas":')[1].split('"edited_code":')[0].strip()
            start_quote = self.detect_quote_type(problem)
            if start_quote is not None:
                problem = problem.replace(start_quote, "")[:-1] # removing ,
            
            code = self.json_str.split('"edited_code":')[1].strip()
            if code.endswith("}"):
                code = code.strip()[:-1].strip()
            
            start_quote = self.detect_quote_type(code)
            if start_quote is not None:
                code = code.replace(start_quote, "")
     
            code = code if code != "" else None
            # code = code_match.group(2) if code_match else None
            
            if problem or code:
                return {
                    "improvement_areas": problem,
                    "edited_code": code
                }
            else:
                raise ValueError("Failed to extract problem or code from malformed JSON")


def get_start_idx(path: str, reverse=False) -> int:
    with open(path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    if not lines:
        return 0
    first_line = lines[0]
    last_line = lines[-1]
    start_index = int(json.loads(first_line)['seed_code_id'])
    last_index = int(json.loads(last_line)['seed_code_id'])
    if reverse:
      return last_index - 1
    else:
      return last_index - start_index + 1

class CodeExtractor:
    def __init__(self):
        self.PROBLEM_START = "###PROBLEM_STATEMENT###"
        self.PROBLEM_END = "###END_PROBLEM_STATEMENT###"
        self.CODE_START = "###ORIGINAL_CODE###"
        self.CODE_END = "###END_ORIGINAL_CODE###"
        self.METADATA_START = "###METADATA###"
        self.METADATA_END = "###END_METADATA###"

        # New delimiters for edits
        self.PREF1_START = "###PREFERRED_SOLUTION_1###"
        self.PREF1_END = "###END_PREFERRED_SOLUTION_1###"
        self.PREF2_START = "###PREFERRED_SOLUTION_2###"
        self.PREF2_END = "###END_PREFERRED_SOLUTION_2###"
        self.NON_PREF_START = "###NON_PREFERRED_SOLUTION###"
        self.NON_PREF_END = "###END_NON_PREFERRED_SOLUTION###"
        self.DIFF_START = "###DIFFERENCES_EXPLAINED###"
        self.DIFF_END = "###END_DIFFERENCES_EXPLAINED###"

        # New delimiters for instructions
        self.DETAILED_START = "###DETAILED_INSTRUCTION###"
        self.DETAILED_END = "###END_DETAILED_INSTRUCTION###"
        self.CONCISE_START = "###CONCISE_INSTRUCTION###"
        self.CONCISE_END = "###END_CONCISE_INSTRUCTION###"
        self.HUMAN_START = "###HUMAN_INSTRUCTION###" 
        self.HUMAN_END = "###END_HUMAN_INSTRUCTION###"
        self.CONV_START = "###CONVERSATIONAL_INSTRUCTION###"
        self.CONV_END = "###END_CONVERSATIONAL_INSTRUCTION###"

        # New delimiters for quality check
        self.COHERENCE_START = "###COHERENCE_CHECK###"
        self.COHERENCE_END = "###END_COHERENCE_CHECK###"
        self.QUALITY_START = "###QUALITY_CHECK###"
        self.QUALITY_END = "###END_QUALITY_CHECK###"
        self.VERDICT_START = "###FINAL_VERDICT###"
        self.VERDICT_END = "###END_FINAL_VERDICT###"

    def _extract_section(self, text, start_delimiter, end_delimiter):
        """Helper method to extract content between delimiters"""
        try:
            start_idx = text.index(start_delimiter) + len(start_delimiter)
            end_idx = text.index(end_delimiter)
            return text[start_idx:end_idx].strip()
        except ValueError as e:
            raise ValueError(f"Failed to find delimiter: {str(e)}")

    def problem_code_extractor(self, text):
        try:
            problem_statement = self._extract_section(text, self.PROBLEM_START, self.PROBLEM_END)
            original_code = self._extract_section(text, self.CODE_START, self.CODE_END)
            metadata = self._extract_section(text, self.METADATA_START, self.METADATA_END)
            
            return {
                'problem_statement': problem_statement,
                'original_code': original_code,
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error extracting content: {str(e)}")
            return None

    def code_edit_extractor(self, text):
        try:
            preferred_1 = self._extract_section(text, self.PREF1_START, self.PREF1_END)
            preferred_2 = self._extract_section(text, self.PREF2_START, self.PREF2_END)
            non_preferred = self._extract_section(text, self.NON_PREF_START, self.NON_PREF_END)
            differences = self._extract_section(text, self.DIFF_START, self.DIFF_END)
            
            return {
                'preferred_edit_a': preferred_1,
                'preferred_edit_b': preferred_2,
                'non_preferred_edit': non_preferred,
                'explanation': differences
            }
            
        except Exception as e:
            print(f"Error extracting edits: {str(e)}")
            return None
    def instructions_extractor(self, text):
        try:
           detailed = self._extract_section(text, self.DETAILED_START, self.DETAILED_END)
           concise = self._extract_section(text, self.CONCISE_START, self.CONCISE_END)
           human = self._extract_section(text, self.HUMAN_START, self.HUMAN_END)
           conversational = self._extract_section(text, self.CONV_START, self.CONV_END)
           
           return {
               'detailed_instruction': detailed,
               'concise_instruction': concise,
               'human_instruction': human,
               'conversational_instruction': conversational
           }
           
        except Exception as e:
          print(f"Error extracting instructions: {str(e)}")
          return None
        
    def _extract_score_explanation(self, section_text, index):
       """Helper to extract score and explanation from a section"""
       lines = section_text.split('\n')
       for i, line in enumerate(lines):
           if line.startswith(f"{index}."):
               # Find score line
               for j in range(i, len(lines)):
                   if lines[j].startswith("Score:"):
                       score = float(lines[j].split(':')[1].strip().strip('[]'))
                       # Find explanation line
                       for k in range(j, len(lines)):
                           if lines[k].startswith("Explanation:"):
                               explanation = lines[k].split(':')[1].strip().strip('[]')
                               return score, explanation
       return None, None

    def quality_check_extractor(self, text):
       try:
           coherence = self._extract_section(text, self.COHERENCE_START, self.COHERENCE_END)
           quality = self._extract_section(text, self.QUALITY_START, self.QUALITY_END)
           verdict = self._extract_section(text, self.VERDICT_START, self.VERDICT_END)
           
           scores = {}
           
           for i in range(1, 3):
               score, explanation = self._extract_score_explanation(coherence, i)
               scores[f'coherence_{i}'] = {
                   'score': score,
                   'explanation': explanation
               }
           
           for i in range(1, 4):
               score, explanation = self._extract_score_explanation(quality, i)
               scores[f'quality_{i}'] = {
                   'score': score,
                   'explanation': explanation
               }
           
           verdict_lines = verdict.split('\n')
           strengths = []
           weaknesses = []
           recommendations = []
           
           current_section = None
           for line in verdict_lines:
               line = line.strip()
               if line.startswith('Strengths:'):
                   current_section = strengths
               elif line.startswith('Weaknesses:'):
                   current_section = weaknesses
               elif line.startswith('Recommendations:'):
                   current_section = recommendations
               elif line.startswith('-') and current_section is not None:
                   current_section.append(line[1:].strip())
           
           return {
               'scores': scores,
               'verdict': {
                   'strengths': strengths,
                   'weaknesses': weaknesses,
                   'recommendations': recommendations
               }
           }
           
       except Exception as e:
           print(f"Error extracting quality check: {str(e)}")
           return None