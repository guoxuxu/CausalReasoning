
import logging
import torch
from pathlib import Path
import time


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


if __name__ == "__main__":
    from utils.model_utils import parse_args, load_model, setup_logging
    from utils.data_utils import load_data, results_save_path, prepare_input
    from transformers import set_seed
    
    setup_logging()
    args = parse_args()
    set_seed (42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    dataset = load_data(args.dataset_name)
    
    Results_ROOT = Path("Results")
    data_save_path = results_save_path(Results_ROOT, args)
    
    model, tokenizer = load_model(args.data_model)
    start_time = time.time()
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    logging.info(f"{start_time_str}: {args.data_model} Generating Data ...")
    
    
    if 'causal' in args.dataset_name:
        from_effect_to_cause_wo = dataset.filter(lambda x: x["Question Type"] == "from effect to cause without intervention")
        from_effect_to_cause_w  = dataset.filter(lambda x: x["Question Type"] == "from effect to cause with intervention")
        from_cause_to_effect_w  = dataset.filter(lambda x: x["Question Type"] == "from cause to effect with intervention")
        from_cause_to_effect_wo = dataset.filter(lambda x: x["Question Type"] == "from cause to effect without intervention")

        problem_count = 0
        for example in from_effect_to_cause_wo:
            input = prepare_input(args.dataset_name, example, args)
            
        
    