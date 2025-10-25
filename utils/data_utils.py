from pathlib import Path
import logging
import os
from datasets import load_dataset
from collections import Counter 
from datasets import Dataset, DatasetDict
import random
from tqdm import tqdm
import json


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
        
    elif dataset_name == 'causal_text':
        dataset = load_dataset("csv", data_files=os.path.join(DATA_PATH, "CausalBench/CausalBench_Text_Part.csv"))
        dataset = dataset.rename_column("Scenario ID", "ID")
        dataset = dataset.rename_column("Scenario and Question", "Problem")
    
    elif dataset_name == 'medmcqa':
        dataset = load_dataset("openlifescienceai/medmcqa")
        dataset["dev"] = dataset["validation"]
        dataset = dataset[data_split]
        filtered = dataset.map(
            lambda example: {"Problem": f"{example.get('question')}\n\nOption 1: {example.get('opa')}\nOption 2: {example.get('opb')}\nOption 3: {example.get('opc')}\nOption 4: {example.get('opd')}\n\nSelect the correct option."}
        )
        filtered = filtered.filter(lambda x: x["exp"] is not None)
        filtered = filtered.rename_column("exp", "Explanation")
        filtered = filtered.rename_column("cop", "Ground Truth")
        filtered = filtered.rename_column("subject_name", "Question Type")
        filtered = filtered.map(
            lambda example, idx: {"ID": idx + 1}, with_indices=True
        )

    elif dataset_name == 'cola':
        dataset = load_dataset("json", data_files=os.path.join(DATA_PATH, "COLA/COPES.json"))
        # filter res_idx is not integer and is null and is less than 0
        dataset = dataset.filter(lambda x: isinstance(x["res_idx"], int) and x["res_idx"] >= 0)
        dataset = dataset.map(
            lambda example: {"Effect": example.get("story")[example.get("res_idx")]}
        )
        # filter rows where cause_idx is not list and is null list
        dataset = dataset.filter(lambda x: isinstance(x["cause_idx"], list) and len(x["cause_idx"]) > 0)
        dataset = dataset.map(
            lambda example: {"Ground Truth": example.get("cause_idx")}
        )
        dataset = dataset.map(
            lambda example: {"Problem": "\n".join([f"Event {i+1}: {s}" for i, s in enumerate(example["story"])])}
        )
        dataset = dataset.map(
            lambda example: {"Problem": example.get("Problem") + f"\n\nEffect: {example.get('Effect')}\n\n" + "Which of the events have a **direct causal link** to the effect? "}
        )
        # add ID column
        filtered = dataset[data_split]
        filtered = filtered.map(
            lambda example, idx: {"ID": idx + 1}, with_indices=True
        )
    
    elif dataset_name == 'copa':
        dataset = load_dataset(
            "json",
            data_files={
                "train": str(DATA_PATH / "COPA/train.jsonl"),
                "dev":   str(DATA_PATH / "COPA/val.jsonl"),
            }
        )
        for split in dataset.keys():
            ds = dataset[split]
            ds = ds.map(lambda ex: {"label": int(ex["label"]) + 1})
            ds = ds.rename_column("label", "Ground Truth")
            ds = ds.rename_column("idx", "ID")
            dataset[split] = ds
        
        # merge train and dev into one dataset called test
        dataset = DatasetDict({
            "train": Dataset.from_dict({key: sum([dataset[split][key] for split in dataset.keys()], []) 
                                   for key in dataset[list(dataset.keys())[0]].column_names})
        })
        assert data_split == 'train'
        filtered = dataset[data_split]
        filtered = filtered.map(
            lambda example: {
                "Problem": (
                    f"Given the premise - {example.get('premise')}\n\n"
                    f"Two possible hypotheses are proposed:\n"
                    f"Hypothesis 1: {example.get('choice1')}\n"
                    f"Hypothesis 2: {example.get('choice2')}\n"
                    f"{example.get('question')}"
                )
            }
        )
        yes_count = sum(1 for x in filtered if x["Ground Truth"] == 1)
        no_count = sum(1 for x in filtered if x["Ground Truth"] == 2)
        print(f"Ground Truth == 1: {yes_count}")
        print(f"Ground Truth == 2 : {no_count}")
        
        
    elif dataset_name == 'e_care':
        cr_dataset = load_dataset(
            "json",
            data_files={
                "train": str(DATA_PATH / "e-CARE/dataset/Causal_Reasoning/train.jsonl"),
                "dev":   str(DATA_PATH / "e-CARE/dataset/Causal_Reasoning/dev.jsonl"),
            }
        )
        def _read_jsonl(path: Path):
            with open(path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        def _norm_id(v):
            try:
                return int(v)
            except Exception:
                return str(v)
        
        train_explain_data = _read_jsonl(DATA_PATH / "e-CARE/dataset/Explanation_Generation/train.jsonl")
        dev_explain_data   = _read_jsonl(DATA_PATH / "e-CARE/dataset/Explanation_Generation/dev.jsonl")

        train_explanation_map = {}
        for example in train_explain_data:
            ex_id = _norm_id(example.get("index"))
            train_explanation_map[ex_id] = example.get("conceptual_explanation", "")
        dev_explanation_map = {}
        for example in dev_explain_data:
            ex_id = _norm_id(example.get("index"))
            dev_explanation_map[ex_id] = example.get("conceptual_explanation", "")

        # sort by index to align
        for split, exp_map in (("train", train_explanation_map), ("dev", dev_explanation_map)):
            ds = cr_dataset[split]
            explanations = []
            for row in ds:
                ex_id = _norm_id(row.get("index"))
                explanations.append(exp_map.get(ex_id, ""))
            
            ds = ds.add_column("Explanation", explanations)
            ds = ds.map(lambda ex: {"label": int(ex["label"]) + 1})
            ds = ds.rename_column("index", "ID")
            ds = ds.rename_column("label", "Ground Truth")  
            cr_dataset[split] = ds
    
        filtered = cr_dataset[data_split]
        filtered = filtered.map(
            lambda example: {
                "Problem": (
                    f"Determine the {example.get('ask-for')} of the premise - {example.get('premise')}\n\n"
                    f"Two possible hypotheses are proposed:\n"
                    f"Hypothesis 1: {example.get('hypothesis1')}\n"
                    f"Hypothesis 2: {example.get('hypothesis2')}\n"
                    "Which hypothesis most accurately reflects the causal relation implied by the premise?"
                )
            }
        )
        yes_count = sum(1 for x in filtered if x["Ground Truth"] == 1)
        no_count = sum(1 for x in filtered if x["Ground Truth"] == 2)
        print(f"Ground Truth == 1: {yes_count}")
        print(f"Ground Truth == 2 : {no_count}")
    
    
    if "causal" in dataset_name:
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


def get_prompt(prompt_name: str) -> str:
    if prompt_name == "causal_map_prompt":
        return (
            "\nYou are an expert in causal reasoning. "
            'Given the following problem description and question, identify all relevant variables and describe their direct causal relationships.'
            'Your output should be a clear causal map in textual form, listing directed causal links as "A → B" (meaning A directly causes B).'
            "Avoid redundancy or duplicate links. List only distinct and meaningful causal relations. "
            'Do not provide any final answer to the question. Focus only on extracting the causal structure. Output strictly in JSON format: {"causal_relation": "A → B; C → D; ..."}.\n'
        )
    elif prompt_name == "causal_integration_prompt":
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
    if ("causal" in dataset_name):
        answer_format = '{"answer":"Yes"} or {"answer":"No"}'
    elif dataset_name == "e_care":
        answer_format = '{"answer": 1} or {"answer": 2}'
    elif dataset_name == "cola":
        answer_format = '{"answer": [event_numbers in order]}. List event numbers in order. (If there is only one cause, still respond as a list)'        
    else:
        answer_format = ''
        
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
        
    elif pm == "zs_causal_Inte_cot":
        causal_map = example.get("causal_map_integration")
        prefix = (
            "\nYou are an expert in causal reasoning. "
            'Given the following problem description, question, and the causal statements related to this problem.'
            'Think step by step before answering. '
            f'After reasoning, output your final answer in the JSON format: {answer_format}. \n\n'
        )
        prompt = prefix + problem  + '\n\n' + causal_map
    else:
        raise ValueError(f"Unsupported prompting method: {prompting_method}")
    
    return prompt


def create_save_path(Results_ROOT, args):
    generation_param = f"temperature{args.temperature}_max_new_tokens{args.max_new_tokens}"
    logging.info(
        "generation_param: %s",
        generation_param,
    )
    data_path = (
        Results_ROOT
        / args.dataset_name
        / args.data_split
        / args.data_model
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
    
    