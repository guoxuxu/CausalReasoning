import logging
import os
import time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from pathlib import Path


def get_api_key():
    return os.getenv("OPENAI_API_KEY")


def call_openai_model(
    prompt,
    model_name="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512,
    num_return_sequences=1,
    log_path=None,
):
    # price per 1M tokens in USD
    # https://platform.openai.com/docs/pricing
    MODEL_PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.00, "output": 60.00},
    }
    client = OpenAI(api_key=get_api_key())
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_return_sequences, 
        )
        
        outputs = [choice.message.content for choice in response.choices]
        
        usage = response.usage
        prompt_toks = usage.prompt_tokens
        completion_toks = usage.completion_tokens
        total_toks = usage.total_tokens
        if model_name in MODEL_PRICES:
            price_info = MODEL_PRICES[model_name]
            cost = (
                (prompt_toks / 1000000) * price_info["input"]
                + (completion_toks / 1000000) * price_info["output"]
            )
        else:
            cost = None
            
        logging.info(
            f"[{model_name}] Prompt tokens: {prompt_toks}, Completion tokens: {completion_toks}, "
            f"Total tokens: {total_toks}, Cost: ${cost:.6f}"
            if cost is not None
            else f"[{model_name}] Token usage: {usage}"
        )
        
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nModel: {model_name}\nPrompt: {prompt}\nOutputs: {outputs}\nCost: {cost}\n")

        return outputs, cost
        
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return [""], 0.0
    


if __name__ == "__main__":
    from utils.model_utils import setup_logging, parse_args
    from utils.data_utils import create_save_path
    from transformers import set_seed
    
    setup_logging()
    args = parse_args()
    set_seed (42)
    
    # dataset_name = args.dataset_name
    # dataset = load_data(DATA_PATH=Path("Data"), dataset_name=dataset_name, data_split=args.data_split)
    
    prompt = "\nYou are an expert in causal reasoning. Given the following problem description and question, identify all relevant variables and describe their direct causal relationships.Your output should be a clear causal map in textual form, listing directed causal links as \"A → B\" (meaning A directly causes B).Avoid redundancy or duplicate links. List only distinct and meaningful causal relations. Do not provide any final answer to the question. Focus only on extracting the causal structure. Output strictly in JSON format: {\"causal_relation\": \"A → B; C → D; ...\"}.\n"
    problem = "def update_scores(scores, bonus_points, student, is_top_student):\n    if is_top_student:\n        scores[student] += (bonus_points * 2)\n    else:\n        scores[student] += bonus_points\n\n    return scores\n\nscores_dict = {'Alice': 90, 'Bob': 85, 'Charlie': 92}\nbonus = 5\nis_top = 'Alice'\nupdate_scores(scores_dict, bonus, is_top, is_top == 'Alice')\",\n\nIf the value of 'bonus_points' is increased, will the scores of all students in 'scores' increase without changing other inputs?"
    prompt = prompt + problem
    answer_format = '{"answer":"Yes"} or {"answer":"No"}'
    
    prefix = (
            "\nYou are given a problem description and a question. "
            f'Answer exactly in the JSON format: {answer_format}. '
            "Do not include any explanations or extra text. \n\n"
        )
    prompt = prefix + problem
    
        
    responses, cost = call_openai_model(
        prompt,
        model_name='gpt-4o-mini',
        temperature=0.7,
        max_tokens=2048,
        num_return_sequences=3,
    )
    
    for i, r in enumerate(responses, 1):
        logging.info(f"[{i}] {r}\n")
        logging.info(f"Estimated cost: ${cost:.6f}")

    # data_path = create_save_path(Path("Results"), args)
    # logging.info(f"Data will be saved to {data_path}")