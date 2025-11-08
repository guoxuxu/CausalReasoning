
import logging
import torch
from pathlib import Path
import time
import re
from tqdm import tqdm
from datasets import Dataset, load_from_disk
from collections import Counter
import json
import os
from openai import OpenAI


def test_generation(prompt, model, tokenizer, args, num_return_sequences=1):
    messages = [
        {"role": "user", "content": prompt}
    ]
    params = dict(conversation=messages, tokenize=False, add_generation_prompt=True)
    try: 
        tokenizer.apply_chat_template.__code__.co_varnames
        if "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames:
            params["enable_thinking"] = False
    except:
        pass
    text = tokenizer.apply_chat_template(**params)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_return_sequences=getattr(args, "num_return_sequences", num_return_sequences),
        do_sample=True,
    )
    # texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    is_qwen = hasattr(model, "config") and isinstance(getattr(model.config, "model_type", ""), str) and "qwen" in model.config.model_type
    all_outputs = []
    inp_len = model_inputs.input_ids.shape[1]
    for i, gen_ids in enumerate(generated_ids):
        output_ids = gen_ids[inp_len:].tolist() 
        if is_qwen:
            try:
                end_think_id = tokenizer.convert_tokens_to_ids("<|endofthink|>")
            except Exception:
                end_think_id = 151668
            try:
                index = len(output_ids) - output_ids[::-1].index(end_think_id)
            except ValueError:
                index = 0

            # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            text = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        all_outputs.append(text)
    return all_outputs


def extract_causal_relation(text):
    if not isinstance(text, str):
        return None
    s = text.strip()
    
    try:
        obj = json.loads(s)
        return obj.get("causal_relation", "")
    except Exception:
        pass
    
    pattern = re.compile(
        r'{\s*"causal_relation"\s*:\s*"(.*?)"\s*}',
        re.IGNORECASE | re.DOTALL
    )
    
    m = pattern.search(s)
    if not m:
        return ""

    return m.group(1).strip()


def extract_answer(text):
    if not isinstance(text, str):
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj.get("answer", None)
    except Exception:
        pass
    
    # search {"answer":"Yes"} or {"answer":"No"}
    EMBEDDED_JSON_RE = re.compile(r'{\s*"answer"\s*:\s*"(Yes|No)"\s*}', re.IGNORECASE)
    m = EMBEDDED_JSON_RE.search(s)
    if not m:
        return None
    
    v = m.group(1)
    if v.lower() == "yes":
        return "Yes"
    elif v.lower() == "no":
        return "No"
    else:
        return None


def majority_voting(answers):
    answers = [str(a).strip().lower() for a in answers if a]
    counts = Counter(answers)
    max_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == max_count]
    first_idx = next(i for i, a in enumerate(answers) if a in candidates)
    majority_answer = answers[first_idx]
    majority_count = max_count
    return majority_answer, majority_count, first_idx


if __name__ == "__main__":
    from utils.model_utils import parse_args, load_model, setup_logging, get_api_key, call_openai_model
    from utils.data_utils import load_data, create_save_path, prepare_input, get_prompt, sample_subset
    from transformers import set_seed
    
    setup_logging()
    args = parse_args()
    set_seed (42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset_name = args.dataset_name
    dataset = load_data(DATA_PATH=Path("Data"), dataset_name=dataset_name, data_split=args.data_split)
    # dataset = sample_subset(dataset, n_per_class=500)
    # dataset = dataset.select(range(5))
    if args.call_gpt:
        model = OpenAI(api_key=get_api_key())
        tokenizer = None
    else:
        model, tokenizer = load_model(args.answer_model)
    
    
    def _generate_causal_map(d, model, tokenizer, args, causal_data_path, prompt_key="causal_map_prompt", map_key="causal_map", given_ids=None):
        if given_ids is not None:
            given_ids = set(given_ids)
            d = d.filter(lambda ex: ex["ID"] in given_ids)
            
        prompt = get_prompt(prompt_key, args.dataset_name)
        for example in tqdm(d, desc="Generating causal maps"):
            ex_id = example.get("ID")
                
            if args.call_gpt:
                max_try = 1
            else:
                max_try = 5
            causal_map = ""
            cost = None
            total_toks = None
            for attempt in range(max_try):
                try:
                    model_input = prompt + "\n\n" + example.get("Problem")
                    if args.call_gpt:
                        outs, cost, total_toks = call_openai_model(
                            model_input,
                            model,
                            model_name=args.probe_model,
                            temperature=args.temperature,
                            max_tokens=args.max_new_tokens,
                            num_return_sequences=1,
                        )
                    else:
                        outs = test_generation(model_input, model, tokenizer, args, num_return_sequences=1)
                    if outs:
                        causal_map = extract_causal_relation(outs[0]) or ""
                    if causal_map:
                        break
                except Exception as e:
                    continue
                
            data = {
                "ID": ex_id,
                "Problem": example.get("Problem"),
                "Explanation": example.get("Explanation"),
                f"{prompt_key}": prompt,
                f"{map_key}": causal_map,
            }
            if cost:
                data["cost_causal_map"] = cost
            if total_toks:
                data["total_tokens_causal_map"] = total_toks
                
            json_path = causal_data_path / f"{ex_id}.json"
            with open(json_path, "w", encoding="utf-8", errors='ignore') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, skipkeys=True)
                
    
    def _integrate_causal_map(d, model, tokenizer, args, causal_data_path, prompt_key="causal_integration_prompt", map_key="causal_map", given_ids=None):
        if given_ids is not None:
            given_ids = set(given_ids)
            d = d.filter(lambda ex: ex["ID"] in given_ids)
            
        prompt = get_prompt(prompt_key, args.dataset_name)
        for example in tqdm(d, desc="Integrating causal map -> statement"):
            ex_id = example.get("ID")
            json_path = causal_data_path / f"{ex_id}.json"
            with open(json_path, "r", encoding="utf-8") as f:
                causal_data = json.load(f)
            causal_map = str(causal_data.get("causal_map", ""))
            if args.call_gpt:
                max_try = 1
            else:
                max_try = 5
            statement = ""
            cost, total_toks = None, None
            for attempt in range(max_try):
                try:
                    model_input = prompt + "\n\n" + example.get("Problem") + "\n\n" + causal_map
                    if args.call_gpt:
                        outs, cost, total_toks = call_openai_model(
                            model_input,
                            model,
                            model_name=args.probe_model,
                            temperature=args.temperature,
                            max_tokens=args.max_new_tokens,
                            num_return_sequences=1,
                        )
                    else:
                        outs = test_generation(model_input, model, tokenizer, args, num_return_sequences=1)
                    if outs:
                        statement = extract_causal_relation(outs[0]) if outs else ""
                    if statement:
                        break
                except Exception:
                    continue
            
            causal_data[prompt_key] = prompt
            causal_data[f"{map_key}_integration"] = statement
            if cost:
                causal_data["cost_integration"] = cost
            if total_toks:
                causal_data["total_tokens_integration"] = total_toks
            
            with open(json_path, "w", encoding="utf-8", errors='ignore') as f:
                json.dump(causal_data, f, ensure_ascii=False, indent=2, skipkeys=True)
                
    
    def _generate_answers(prompt, model, tokenizer, args, causal_data_path, data_save_path):
        cost, total_toks = None, None
        if args.call_gpt:
            model_outputs, cost, total_toks = call_openai_model(
                prompt,
                model,
                model_name=args.answer_model,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
                num_return_sequences=args.num_return_sequences,
            )
        else:
            model_outputs = test_generation(prompt, model, tokenizer, args)
        
        valid_model_outputs = []
        model_answers = []
        for text in model_outputs:
            answer = extract_answer(text)
            if answer is not None:  
                valid_model_outputs.append(text)
                model_answers.append(answer)
        if model_answers:
            if len(model_answers) > 1:
                final_answer, majority_count, first_idx = majority_voting(model_answers)
            else:
                final_answer = model_answers[0]
        else:
            final_answer = None
        
        return valid_model_outputs, model_answers, final_answer, cost, total_toks
        
    
    def _check_causal_map_files(dataset, causal_data_path: Path, attribute_name:str, overwrite=False):
        missing_ids = []
        for ex in tqdm(dataset, desc="Checking causal map files"):
            ex_id = ex.get("ID")
            if overwrite:
                missing_ids.append(ex_id)
                continue
            json_path = causal_data_path / f"{ex_id}.json"
            if not json_path.exists():
                missing_ids.append(ex_id)
            else:
                with open(json_path, "r", encoding="utf-8") as f:
                    causal_data = json.load(f)
                if causal_data.get(attribute_name, ""):
                    continue
                else:
                    missing_ids.append(ex_id)
        return missing_ids
    
    
    causal_data_path = create_save_path(Path(args.map_path), args.probe_model, args)
    if args.generate_map:
        causal_map_missing_ids = _check_causal_map_files(dataset, causal_data_path, attribute_name="causal_map", overwrite=args.overwrite)
        if causal_map_missing_ids:
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info(f"{start_time_str}: {args.probe_model} Generating Causal Maps for {len(causal_map_missing_ids)} examples ...")
            _generate_causal_map(dataset, model, tokenizer, args, causal_data_path, prompt_key="causal_map_prompt", map_key="causal_map", given_ids=causal_map_missing_ids)
        
    if args.integrate_map:
        causal_map_missing_ids = _check_causal_map_files(dataset, causal_data_path, attribute_name="causal_map_integration", overwrite=args.overwrite)
        if causal_map_missing_ids:
            start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logging.info(f"{start_time_str}: {args.probe_model} Integrating Causal Maps for {len(causal_map_missing_ids)} examples ...")
            _integrate_causal_map(dataset, model, tokenizer, args, causal_data_path, prompt_key="causal_integration_prompt", map_key="causal_map", given_ids=causal_map_missing_ids)

    
    # Evaluation: Generate Answers
    data_save_path = create_save_path(Path(args.results_path), args.answer_model, args)
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"{start_time_str}: {args.answer_model} Generating Answer ...")
    
    evaluation_methods = [
        "zs", "zs_cot", "zs_causal", "zs_causal_Inte", "zs_Explanation", 
    ]
    
    correct_by_method = {m: 0 for m in evaluation_methods}

    for example in tqdm(dataset, desc="Evaluating"):
        ex_id = example.get("ID")
        ground_truth = example["Ground Truth"]
        output_file = data_save_path / f"{ex_id}.json"
        # check if already exists
        if os.path.isfile(output_file) and os.path.getsize(output_file) > 0:
            # check is not null
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:    
            results = {
                "ID": ex_id,
                "Problem": example.get("Problem"),
                'Question Type': example.get("Question Type"),
                "Ground Truth": ground_truth,
            }
        if args.use_map:
            json_path = causal_data_path / f"{ex_id}.json"
            with open(json_path, "r", encoding="utf-8") as f:
                causal_data = json.load(f)
            
            causal_map = str(causal_data.get("causal_map", ""))
            causal_map_integration = str(causal_data.get("causal_map_integration", ""))
            example.update({
                "causal_map": causal_map,
                "causal_map_integration": causal_map_integration,
            })
            results.update({
                "causal_map": causal_map,
                "causal_map_integration": causal_map_integration,
            })
        
        results.update({
            "Explanation": example.get("Explanation"),
        })
        
        prompts = {m: prepare_input(example, m, dataset_name) for m in evaluation_methods}
        _generate_answers_args = [(m, prompts[m]) for m in evaluation_methods]
        
        for method, prompt in _generate_answers_args:
                
            valid_model_outputs, model_answers, final_answer, cost, total_toks = _generate_answers(prompt, model, tokenizer, args, causal_data_path, data_save_path)
            is_acc = (final_answer is not None) and (str(final_answer).lower() == str(ground_truth).lower().strip(""))
            
            results.update({
                f"{method}_prompt": prompt,
                f"{method}_outputs": valid_model_outputs,
                f"{method}_answers": model_answers,
                f"{method}_final_ans": final_answer,
                f"{method}_is_acc": is_acc,
            })
            if cost:
                results[f"{method}_cost"] = cost
            if total_toks:
                results[f"{method}_total_tokens"] = total_toks
            if is_acc:
                correct_by_method[method] += 1
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    N = len(dataset)
    for m, c in correct_by_method.items():
        logging.info(f"{m} Accuracy: {c / N:.2f}")
