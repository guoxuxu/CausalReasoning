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
    problem_load_path = create_save_path(Path("Results"), args)
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"{start_time_str}: {args.data_model} Generating Answer ...")
    
    inpsection_ids_1, inpsection_ids_2, inpsection_ids_3, inpsection_ids_4= [], [], [], []

    evaluation_methods = ["zs", 
                            "zs_cot", 
                            "zs_Explanation", 
                            "zs_causal", 
                            "zs_causal_Inte"]
    valid_types = [
        'from effect to cause without intervention',
        'from effect to cause with intervention',
        'from cause to effect with intervention',
        'from cause to effect without intervention'
    ]
    
    correct_by_method = {m: 0 for m in evaluation_methods}
    correct_by_method_qtyep = {m: {t: 0 for t in valid_types} for m in evaluation_methods}
    
    problem_files = [p for p in problem_load_path.iterdir() if p.suffix == ".json"]
    for problem_file in tqdm(problem_files, desc="Evaluating"):
        with open(problem_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        ex_id = results.get("ID")
        
        for method in evaluation_methods:
            is_acc = results.get(f"{method}_is_acc")
            question_type = results.get("Question Type")
            if is_acc:
                correct_by_method[method] += 1
                correct_by_method_qtyep[method][question_type] += 1
        
        if results["zs_Explanation_is_acc"] == True and results["zs_causal_is_acc"] == False and results["zs_causal_Inte_is_acc"] == False:
            inpsection_ids_1.append(ex_id)
            
        if results["zs_Explanation_is_acc"] == True and results["zs_causal_is_acc"] == True and results["zs_causal_Inte_is_acc"] == False:
            inpsection_ids_2.append(ex_id)
        
        if results["zs_Explanation_is_acc"] == True and results["zs_causal_is_acc"] == False and results["zs_causal_Inte_is_acc"] == True:
            inpsection_ids_3.append(ex_id)

    
    N = len(problem_files)
    logging.info(f"Number of Problems: {N}")
    for m, c in correct_by_method.items():
        logging.info(f"{m} Accuracy: {c / N:.2f}")
    

    logging.info("Question Types: " + "\t".join(valid_types))
    for m, qtyep_dict in correct_by_method_qtyep.items():
        accuracy_list = []
        for t in valid_types:
            qtyep_count = sum(1 for p in problem_files if json.load(open(p, "r", encoding="utf-8")).get("Question Type") == t)
            if qtyep_count > 0:
                accuracy = qtyep_dict[t] / qtyep_count
            else:
                accuracy = 0.0
            accuracy_list.append(f"{accuracy:.2f}")
        logging.info(f"{m} Accuracy: \t\t" + "\t".join(accuracy_list))
    
    logging.info(f"Explanation Correct, causal Wrong: {inpsection_ids_1}")
    logging.info(f"Explanation Correct, causal link Correct, causal integration Wrong: {inpsection_ids_2}")
    logging.info(f"Explanation Correct, causal link Wrong, causal integration Correct: {inpsection_ids_3}")