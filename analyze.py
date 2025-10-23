


if __name__ == "__main__":
    from utils.model_utils import parse_args, load_model, setup_logging
    from utils.data_utils import load_data, create_save_path, prepare_input, get_prompt, sample_subset
    from transformers import set_seed
    
    setup_logging()

    # Evaluation: Check Performance
    data_save_path = create_save_path(Path("Results"), args)
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"{start_time_str}: {args.data_model} Generating Answer ...")
    
    evaluation_methods = ["zs", 
                            "zs_cot", 
                            "zs_Explanation", 
                            "zs_causal", 
                            "zs_causal_Inte"]
    correct_by_method = {m: 0 for m in evaluation_methods}

    for example in tqdm(dataset, desc="Evaluating"):
        ex_id = example.get("ID")
        json_path = data_save_path / f"{ex_id}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        for method in evaluation_methods:
            is_acc = results.get(f"{method}_is_acc")
            if is_acc:
                correct_by_method[method] += 1
        
            