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
                (prompt_toks / 1000) * price_info["input"]
                + (completion_toks / 1000) * price_info["output"]
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
    from utils.model_utils import setup_logging, parse_arguments
    from utils.data_utils import create_save_path
    
    setup_logging()
    args = parse_args()
    set_seed (42)
    
    MODEL_PRICES = {
        "gpt-4o-mini": {"input": 0.000150, "output": 0.000600},
        "gpt-4o": {"input": 0.00250, "output": 0.01000},
        "o1-mini": {"input": 0.0006, "output": 0.0024},
        "o1": {"input": 0.006, "output": 0.024},
    }
        
    
    prompt = "Explain the causal effect of study hours on exam performance in one sentence."

    responses, cost = call_openai_model(
        prompt,
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=256,
        num_return_sequences=3,
    )
    
    for i, r in enumerate(responses, 1):
    logging.info(f"[{i}] {r}\n")
    logging.info(f"Estimated cost: ${cost:.6f}")

    data_path = create_save_path(Path("Data"), args)
    logging.info(f"Data will be saved to {data_path}")