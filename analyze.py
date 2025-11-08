from pathlib import Path
import logging
import time
import json
from tqdm import tqdm


if __name__ == "__main__":
    from utils.model_utils import setup_logging, parse_args
    from utils.data_utils import create_save_path
    
    setup_logging()
    args = parse_args()

    # Evaluation: Check Performance
    problem_load_path = create_save_path(Path(args.results_path), args.answer_model, args)
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"{start_time_str}:  Analyze {args.answer_model} Results ...")
    
    # if not args.call_gpt:
    #     model, tokenizer = load_model(args.answer_model)
    
    inpsection_ids_1, inpsection_ids_2, inpsection_ids_3, inpsection_ids_4= [], [], [], []

    evaluation_methods = ["zs", 
                            "zs_cot", 
                            "zs_causal", 
                            "zs_causal_Inte", "zs_Explanation"]
    if args.dataset_name != "cola":
        evaluation_methods.extend(["zs_Explanation", ])
    valid_types = [
        'from effect to cause without intervention',
        'from effect to cause with intervention',
        'from cause to effect without intervention',
        'from cause to effect with intervention',
    ]
    qtyep_count_dict = {q: 0 for q in valid_types}
    
    correct_by_method = {m: 0 for m in evaluation_methods}
    correct_by_method_qtyep = {m: {t: 0 for t in valid_types} for m in evaluation_methods}
    pass_1_correct_by_method = {m: 0 for m in evaluation_methods}
    pass_1_correct_by_method_qtyep = {m: {t: 0 for t in valid_types} for m in evaluation_methods}
    
    problem_files = [p for p in problem_load_path.iterdir() if p.suffix == ".json"]
    problem_count = 0
    inpsection_ids = []
    cot_lengths = []
    cot_sc_lengths = []
    causal_lengths = []
    causal_Inte_lengths = []
    explanation_lengths = []
    for problem_file in tqdm(problem_files, desc="Evaluating"):
        with open(problem_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        ex_id = results.get("ID")
        ground_truth = str(results["Ground Truth"]).lower().strip()
        question_type = results.get("Question Type", "")
        
        problem_count += 1
        
        for t in valid_types:
            if question_type == t:
                qtyep_count_dict[t] += 1
        
        # calculate output length  
        sc_len = []
        for cot_out in results.get(f"zs_cot_outputs"):
            cot_length = len(cot_out.split(" ")) if cot_out else 0
            sc_len.append(cot_length)
        cot_sc_length = sum(sc_len) / len(sc_len) if sc_len else 0
        cot_sc_lengths.append(cot_sc_length)
        
        cot_length = len(results.get(f"zs_cot_outputs")[0].split(" ")) if results.get(f"zs_cot_outputs") else 0
        causal_Inte_length = len(results.get(f"causal_map_integration").split(" ")) if results.get(f"causal_map_integration") else 0
        causal_length = len(results.get(f"causal_map").split(" ")) if results.get(f"causal_map") else 0
        explanation_length = len(results.get(f"Explanation").split(" ")) if results.get(f"Explanation") else 0
        cot_lengths.append(cot_length)
        causal_lengths.append(causal_length)
        causal_Inte_lengths.append(causal_Inte_length)
        explanation_lengths.append(explanation_length)
        
        for method in evaluation_methods:
            is_acc = results.get(f"{method}_is_acc")
            if is_acc == True:
                correct_by_method[method] += 1
                if question_type in valid_types:
                    correct_by_method_qtyep[method][question_type] += 1
        
        for method in evaluation_methods:
            try:
                pass_1_ans = results.get(f"{method}_answers")[0].lower().strip()
            except:
                pass_1_ans = None
            
            is_acc = (pass_1_ans is not None) and (pass_1_ans == ground_truth)
            if is_acc:
                pass_1_correct_by_method[method] += 1
                if question_type in valid_types:
                    pass_1_correct_by_method_qtyep[method][question_type] += 1
        
    
    # average lengths
    avg_cot_length = sum(cot_lengths) / len(cot_lengths) if cot_lengths else 0
    avg_sc_length = sum(cot_sc_lengths) / len(cot_sc_lengths) if cot_sc_lengths else 0
    avg_causal_length = sum(causal_lengths) / len(causal_lengths) if causal_lengths else 0
    avg_causal_Inte_length = sum(causal_Inte_lengths) / len(causal_Inte_lengths) if causal_Inte_lengths else 0
    avg_explanation_length = sum(explanation_lengths) / len(explanation_lengths) if explanation_lengths else 0
    logging.info(f"Average CoT Length: {avg_cot_length:.2f}")
    logging.info(f"Average SC Length: {avg_sc_length:.2f}")
    logging.info(f"Average Causal Map Length: {avg_causal_length:.2f}")
    logging.info(f"Average Causal Integration Length: {avg_causal_Inte_length:.2f}")
    logging.info(f"Average Explanation Length: {avg_explanation_length:.2f}")
    
    N = problem_count
    logging.info(f"Number of Problems: {N}")
    if N == 0:
        logging.info("No problems found. Exiting.")
        exit(0)
    logging.info("Pass-1 Correct: ===========================")
    for m, c in pass_1_correct_by_method.items():
        logging.info(f"{m} Pass-1 Accuracy: \t\t{(c / N) * 100:.2f} ({c}/{N})")
    logging.info("SC Correct: ===========================")
    for m, c in correct_by_method.items():
        logging.info(f"{m} Accuracy: \t\t{(c / N) * 100:.2f}({c}/{N})")
    
    
    logging.info("Question Types: " + "\t".join(valid_types))
    qtyep_count_list = [str(qtyep_count_dict[t]) for t in valid_types]
    logging.info("Question Type Counts:  \t\t\t" + "\t".join(qtyep_count_list))
    
    logging.info("Pass-1 Correct by Question Type: ===========================")
    for m, qtyep_dict in pass_1_correct_by_method_qtyep.items():
        accuracy_list = []
        for t in valid_types:
            qtyep_count = qtyep_count_dict[t]
            if qtyep_count > 0:
                accuracy = qtyep_dict[t] / qtyep_count
            else:
                accuracy = 0.0
            accuracy_list.append(f"{accuracy * 100:.2f}")
        logging.info(f"{m} Accuracy: \t\t\t" + "\t".join(accuracy_list))

    logging.info("SC Correct by Question Type: ===========================")
    for m, qtyep_dict in correct_by_method_qtyep.items():
        accuracy_list = []
        for t in valid_types:
            qtyep_count = qtyep_count_dict[t]
            if qtyep_count > 0:
                accuracy = qtyep_dict[t] / qtyep_count
            else:
                accuracy = 0.0
            accuracy_list.append(f"{accuracy * 100:.2f}")
        logging.info(f"{m} Accuracy: \t\t\t" + "\t".join(accuracy_list))

        