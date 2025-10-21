from math_verify import parse, verify
from collections import Counter
from collections import defaultdict
import numpy as np
import itertools
import json

def normalize_answer(a: str) -> str:
    """Normalize model answers for fair majority voting."""
    if not isinstance(a, str):
        a = str(a)
    a = a.strip()
    try:
        num = float(a)
        if num.is_integer():
            return str(int(num))
        else:
            return str(round(num, 2))
    except ValueError:
        return a.rstrip(".")
    

def majority_voting(answers):
    answers = [normalize_answer(a) for a in answers if a]
    counts = Counter(answers)
    max_count = max(counts.values())
    candidates = [a for a, c in counts.items() if c == max_count]
    first_idx = next(i for i, a in enumerate(answers) if a in candidates)
    majority_answer = answers[first_idx]
    majority_count = max_count
    return majority_answer, majority_count, first_idx


def desceding_drop_slope(y):
    if len(y) <= 1:
        return 0.0
    x = np.arange(len(y), dtype=float)
    y = np.asarray(y, dtype=float)
    
    m = np.isfinite(y)
    if m.sum() <= 1:
        return 0.0
    x, y = x[m], y[m]
    
    # y0 = y[0]
    # indices = [i for i, val in enumerate(y) if val < y0]
    # values = [y[i] for i in indices]
    # if len(values) == 0:
    #     return 0.0
    # if len(values) == 1:
    #     return (values[0] - y0) / (indices[0] - 0)
    # x = np.array(indices, dtype=float)
    # y_vals = np.array(values, dtype=float)
    
    slope, intercept = np.polyfit(x, y, 1)
    return slope 


def minmax_metric(answers, slopes, lengths, w=(1.0, 1.0, 1.0), count_bonus_factor=0.85, eps=1e-12):
    answers = [normalize_answer(a) for a in answers if a]
    slopes = np.asarray(slopes, float)
    lengths= np.asarray(lengths, float)
    
    idx_map = defaultdict(list)
    for i, ans in enumerate(answers):
        idx_map[ans].append(i)
    uniq_answers = list(idx_map.keys())
    
    mean_slope, mean_length, counts = [], [], []
    for ans in uniq_answers:
        idxs = np.array(idx_map[ans], dtype=int)
        mean_slope.append(  slopes[idxs].mean()  )
        mean_length.append( lengths[idxs].mean() )
        counts.append(  len(idxs) )

    mean_slope  = np.array(mean_slope,  dtype=float)
    mean_length = np.array(mean_length, dtype=float)
    counts      = np.array(counts,      int)

    xs = -mean_slope
    xs = (xs - xs.min()) / (xs.max() - xs.min() + eps)
    loss_slope = 1.0 - xs

    xl = (mean_length - mean_length.min()) / (mean_length.max() - mean_length.min() + eps)
    loss_len = xl
    
    lc = np.log1p(counts)  #
    lc_norm = (lc - lc.min()) / (lc.max() - lc.min() + eps)
    loss_count = 1.0 - lc_norm   #

    score = w[0]*loss_slope + + w[1]*loss_len + w[2] * loss_count
    score = np.where(counts >= 2, score * float(count_bonus_factor), score)
    
    best_idx = int(np.argmin(score))
    best_answer = uniq_answers[best_idx]
    return idx_map[best_answer][0]

def slope_and_length(answers, slopes, lengths, w=(1.0, 1.0, 1.0), count_bonus_factor=0.85, eps=1e-12):
    answers = [normalize_answer(a) for a in answers if a]
    slopes = np.asarray(slopes, float)
    lengths= np.asarray(lengths, float)
    
    xs = -slopes
    xs = (xs - xs.min()) / (xs.max() - xs.min() + eps)
    loss_slope = 1.0 - xs

    xl = (lengths - lengths.min()) / (lengths.max() - lengths.min() + eps)
    loss_len = xl

    score = w[0]*loss_slope + + w[1]*loss_len 
    best_idx = int(np.argmin(score))
    return best_idx


def random_search(model_nn_key, solution_files):
    choices = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    grid = list(itertools.product(choices, choices, choices))
    all_correct_num = []
    for combination in grid:
        slope_weight, length_weight, count_weight = combination
        ig_correct = 0
        for pf in solution_files:
            try:
                with pf.open("r", encoding="utf-8") as f:
                    example = json.load(f)
            except Exception as e:
                continue
            
            if model_nn_key not in example:
                continue
            
            model_solutions = example["model_output"]
            sc_answers = example["sc_answers"]
            
            sc_lengths = [len(s.split("\n\n")) for s in model_solutions]
            slopes = np.array([desceding_drop_slope(y) for y in example[model_nn_key]], dtype=float)
            ig_pred_index = minmax_metric(
                sc_answers, slopes, sc_lengths,
                w=(slope_weight, length_weight, count_weight)
            )
            ig_is_correct = verify(parse(model_solutions[ig_pred_index]), parse(example['solution']))
            if ig_is_correct:
                ig_correct += 1
        all_correct_num.append(ig_correct)
    all_correct_num = np.array(all_correct_num)
    best_idx = all_correct_num.argmax()
    return grid[best_idx]
            