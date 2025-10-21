from pathlib import Path
import logging
import os
from datasets import load_dataset
from collections import Counter 
from datasets import Dataset, DatasetDict


def load_data(dataset_name):
    if dataset_name.lower() == 'causal_code':
        dataset = load_dataset("csv", data_files="./CausalBench/CausalBench_Code_Part.csv")
    elif dataset_name.lower() == 'causal_math':
        dataset = load_dataset("csv", data_files="./CausalBench/CausalBench_Math_Part.csv")
    elif dataset_name.lower() == 'causal_text':
        dataset = load_dataset("csv", data_files="./CausalBench/CausalBench_Text_Part.csv")
            
    train_ds = dataset["train"]
    filtered = train_ds.filter(lambda x: x["Ground Truth"] in ["Yes", "No"])
    yes_count = sum(1 for x in filtered if x["Ground Truth"] == "Yes")
    no_count = sum(1 for x in filtered if x["Ground Truth"] == "No")
    print(f"Ground Truth == 'Yes': {yes_count}")
    print(f"Ground Truth == 'No' : {no_count}")
    filtered = filtered.map(lambda x: {"Question Type": x["Question Type"].lower() if isinstance(x["Question Type"], str) else x["Question Type"]})
    valid_types = {
        'from effect to cause without intervention',
        'from effect to cause with intervention',
        'from cause to effect with intervention',
        'from cause to effect without intervention'
    }
    filtered = filtered.filter(lambda x: x["Question Type"] in valid_types)
    print(Counter(filtered['Question Type']))
    return filtered


def build_prompt_row(example, dataset_name: str, prompting_method: str) -> str:
    dn = (dataset_name or "").lower()
    pm = (prompting_method or "").lower().strip()
    
    if dn == "causal_code":
        base = example.get("Code") + example.get("Question")
    elif dn == "causal_math":
        base = example.get("Mathematical Scenario") + example.get("Question")
    elif dn == "causal_text":
        base = example.get("Scenario and Question")
    else:
        base = example.get("Question")
        
    explanation = example.get("Explanation")
    # question_type = example.get("Question Type")
    # ground_truth = example.get("Ground Truth")
    
    if pm == "zero_shot_insert_explanations":
        suffix = (
            "\nYou are given a problem description, a yes/no question, and an explanation. "
            'Answer exactly in the JSON format: {"answer":"Yes"} or {"answer":"No"}. '
            "Do not include any explanations or extra text."
        )
        prompt = base + ("\n" + explanation if explanation else "") + suffix
    elif pm == "zero_shot_cot":
        suffix = (
            "\nYou are given a problem description and a yes/no question. "
            "Think step by step before answering. "
            'After reasoning, output your final answer in the JSON format: {"answer":"Yes"} or {"answer":"No"}.'
        )
        prompt = base + suffix
    else:
        # default to zero_shot
        suffix = (
            "\nYou are given a problem description and a yes/no question. "
            'Answer exactly in the JSON format: {"answer":"Yes"} or {"answer":"No"}. '
            "Do not include any explanations or extra text."
        )
        prompt = base + suffix
    return prompt


def add_prompt_column(ds, dataset_name: str, prompting_method: str):
    def _mapper(example):
        return {"prompt": build_prompt_row(example, dataset_name, prompting_method)}

    if isinstance(ds, DatasetDict):
        return ds.map(_mapper)
    else:
        return ds.map(_mapper)
    

def prepare_input(dataset_name, dataset, prompting_method):
    dataset = add_prompt_column(dataset, dataset_name=dataset_name, prompting_method=prompting_method)
    return dataset


def results_save_path(Results_ROOT, args):
    args.generation_param = f"temperature{args.temperature}_max_new_tokens{args.max_new_tokens}"
    logging.info(
        "Data generation: %s | hyperparameter: %s",
        args.prompting_method,
        args.generation_param,
    )
    data_save_path = (
        Results_ROOT
        / args.dataset_name
        / args.data_split
        / args.data_model
        / args.prompting_method
        / args.generation_param
        / args.category
    )
    if not os.path.exists(data_save_path):
        data_save_path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    from model_utils import parse_args
    from model_utils import setup_logging
    setup_logging()
    args = parse_args()
    
    dataset = load_data("causal_code")
    
    Results_ROOT = Path("Results")
    data_save_path = results_save_path(Results_ROOT, args)
    logging.info(
        "Data will be saved to %s",
        data_save_path
    )
    
    