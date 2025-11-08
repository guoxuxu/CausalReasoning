from pathlib import Path
import logging
import os
from datasets import load_dataset
from collections import Counter 
from datasets import Dataset, DatasetDict
import random
from tqdm import tqdm
import json
from datasets import concatenate_datasets


def load_data(DATA_PATH, dataset_name, data_split):
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'causal_code':
        dataset = load_dataset("csv", data_files=os.path.join(DATA_PATH, "CausalBench/CausalBench_Code_Part.csv"))
        dataset = dataset.rename_column("Causal_Scenario_ID", "ID")
        dataset = dataset.map(
            lambda example: {"Problem": example.get("Code") + "\n\n" + example.get("Question")}
        )
        
    elif dataset_name == 'causal_math':
        dataset = load_dataset("csv", data_files=os.path.join(DATA_PATH, "CausalBench/CausalBench_Math_Part.csv"))
        dataset = dataset.rename_column("Scenario_ID", "ID")
        dataset = dataset.map(
            lambda example: {"Problem": example.get("Mathematical Scenario") + "\n\n" + example.get("Question")}
        )
    else:
        raise NotImplementedError
    
    dataset = dataset[data_split]
    filtered = dataset.filter(lambda x: x["Ground Truth"] in ["Yes", "No"])
    filtered = filtered.map(lambda x: {"Question Type": x["Question Type"].lower() if isinstance(x["Question Type"], str) else x["Question Type"]})
    valid_types = [
        'from effect to cause without intervention',
        'from effect to cause with intervention',
        'from cause to effect with intervention',
        'from cause to effect without intervention'
    ]
    filtered = filtered.filter(lambda x: x["Question Type"] in valid_types)
    yes_count = sum(1 for x in filtered if x["Ground Truth"] == "Yes")
    no_count = sum(1 for x in filtered if x["Ground Truth"] == "No")
    print(f"Ground Truth == 'Yes': {yes_count}")
    print(f"Ground Truth == 'No' : {no_count}")
    print(Counter(filtered['Question Type']))
    filtered = filtered.map(lambda example, idx: {"ID": idx + 1}, with_indices=True)
    return filtered


def sample_subset(dataset, n_per_class=500):
    rng = random.Random(42)
    valid_types = {
        'from effect to cause without intervention',
        'from effect to cause with intervention',
        'from cause to effect with intervention',
        'from cause to effect without intervention'
    }
    sampled_datasets = []
    for qtype in valid_types:
        subset = dataset.filter(lambda x: x["Question Type"] == qtype)
        n = min(len(subset), n_per_class)
        indices = rng.sample(range(len(subset)), n)
        sampled = subset.select(indices)
        sampled_datasets.append(sampled)
    final_dataset = Dataset.from_dict({key: sum([d[key] for d in sampled_datasets], []) 
                                   for key in sampled_datasets[0].column_names})
    print("Class distribution:")
    print(Counter(final_dataset["Question Type"]))
    return final_dataset


def get_prompt(prompt_name: str, dataset_name: str) -> str:
    if prompt_name == "causal_map_prompt":
        if dataset_name == "cola":
            return (
                "\nYou are an expert in causal reasoning. "
                "Given the following short story consisting of sequential events leading to an effect, "
                "identify all variables that are factual conditions or mechanisms that directly trigger the final effect."
                "For each direct causal link, ensure one variable triggers another through a factual condition or mechanism."
                'Describe the valid causal link as "A → B" (meaning A directly causes B).'
                "Avoid duplicate, contradictory / cyclic, self-referential, or suprious links. "
                'Your output should be a clear causal map in textual form. '
                'Do not provide any final answer to the question. Focus only on extracting the causal structure. Output strictly in JSON format: {"causal_relation": "A → B; C → D; ..."}.\n'
            )
        else:
            return (
                "\nYou are an expert in causal reasoning. "
                'Given the following problem description and question, identify all relevant variables and describe their direct causal relationships.'
                'Your output should be a clear causal map in textual form, listing directed causal links as "A → B" (meaning A directly causes B).'
                "Avoid redundancy or duplicate links. List only distinct and meaningful causal relations. "
                'Do not provide any final answer to the question. Focus only on extracting the causal structure. Output strictly in JSON format: {"causal_relation": "A → B; C → D; ..."}.\n'
            )
    elif prompt_name == "causal_integration_prompt":
        if dataset_name == "cola":
            return (
                "\nYou are an expert in causal reasoning. "
                'Given the following short story consisting of sequential events leading to an effect, and optionally a list of direct causal links, '
                'If no causal links are provided, first identify all factual and mechanical causal links as "A → B" (meaning A directly causes B). '
                "For each direct causal link, specify a short mechanistic explanation. "
                "Then summarize the verified causal relations into concise natural-language sentences.\n"
                "Use as few sentences as needed to maintain clarity. "
                'Do not provide any final answer to the question. Output strictly in JSON format: {"causal_relation": "<concise natural-language causal description>"}.\n'
            )
        else:
            return (
                "\nYou are an expert in causal reasoning. "
                'Given the following problem description and question, and optionally a list of causal links, '
                'If no causal links are provided, first identify all relevant variables and their direct causal links as "A → B" (meaning A directly causes B). '
                "Examine the causal links to ensure logical consistency: remove duplicate, redundant, or self-referential relations (e.g., both 'A → B' and 'B → A', or 'A → A')."
                "Then, synthesize these causal links into concise, natural-language statements, summarizing how one variable causally influence another."
                "Use as few sentences as needed to maintain clarity. "
                'Do not provide any final answer to the question. Output strictly in JSON format: {"causal_relation": "<concise natural-language causal description>"}.\n'
            )


def prepare_input(example, prompting_method, dataset_name):
    pm = prompting_method
    
    problem = example.get("Problem")
    answer_format = '{"answer":"Yes"} or {"answer":"No"}'
        
    if pm == "zs":
        # default to zero_shot
        prefix = (
            "\nYou are given a problem description and a question. "
            f'Answer exactly in the JSON format: {answer_format}. '
            "Do not include any explanations or extra text. \n\n"
        )
        prompt = prefix + problem
        
            
    elif pm == "zs_cot":
        prefix = (
            "\nYou are given a problem description and a question. "
            "Think step by step before answering. "
            f'After reasoning, output your final answer in the JSON format: {answer_format}. \n\n'
        )
        prompt = prefix + problem
        
    elif pm == "zs_Explanation":
        prefix = (
            "\nYou are given a problem description, a question, and an explanation. "
            f'Answer exactly in the JSON format: {answer_format}. '
            "Do not include any explanations or extra text. \n\n"
        )
        prompt = prefix + problem + "\n\n" + example["Explanation"]
    
    elif pm == "zs_causal":
        causal_map = example.get("causal_map")
        prefix = (
            "\nYou are an expert in causal reasoning. "
            'Given the following problem description, question, and the causal relationships related to this problem.'
            f'Answer exactly in the JSON format: {answer_format}. '
            "Do not include any explanations or extra text. \n\n"
        )
        prompt = prefix  + problem  + '\n\n' + causal_map
    
        
    elif pm == "zs_causal_Inte":
        causal_map = example.get("causal_map_integration")
        prefix = (
            "\nYou are an expert in causal reasoning. "
            'Given the following problem description, question, and the causal statements related to this problem.'
            f'Answer exactly in the JSON format: {answer_format}. '
            "Do not include any explanations or extra text. \n"
        )
        prompt = prefix + problem  + '\n\n' + causal_map
        
    elif pm == "zs_causal_cot":
        causal_map = example.get("causal_map")
        prefix = (
            "\nYou are an expert in causal reasoning. "
            'Given the following problem description, question, and the causal relationships related to this problem.'
            'Think step by step before answering. '
            f'After reasoning, output your final answer in the JSON format: {answer_format}. \n\n'
        )
        prompt = prefix + problem  + '\n\n' + causal_map
        
    else:
        raise ValueError(f"Unsupported prompting method: {prompting_method}")
    
    return prompt


def create_save_path(Results_ROOT, model_name, args):
    generation_param = f"temperature{args.temperature}_max_new_tokens{args.max_new_tokens}"
    logging.info(
        "generation_param: %s",
        generation_param,
    )
    data_path = (
        Results_ROOT
        / args.dataset_name
        / args.data_split
        / model_name
        / generation_param
    )
    if not os.path.exists(data_path):
        data_path.mkdir(parents=True, exist_ok=True)
        
    return data_path


if __name__ == "__main__":
    from model_utils import parse_args
    from model_utils import setup_logging
    setup_logging()
    args = parse_args()
    
    dataset = load_data(DATA_PATH=Path("Data"), dataset_name=args.dataset_name, data_split=args.data_split)
    logging.info(
        "Data columns %s",
        dataset.column_names
    )
    Results_ROOT = Path("Results")
    data_save_path = create_save_path(Results_ROOT, args)
    logging.info(
        "Data will be saved to %s",
        data_save_path
    )
    
    