import argparse
import torch
import logging
from pathlib import Path
from typing import Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
load_dotenv()
import os


SUPPORTED_MODELS = {
    "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
    "qwen2.5-32b": "Qwen2.5-32B",
    "llama3.1-8b-instruct": "llama3.1-8b-instruct",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-5": "gpt-5",
    "o1-mini": "o1-mini",
    "o1": "o1",
}
WEIGHTS_ROOT = Path("/mnt/hdd/weights")


def get_api_key():
    return os.getenv("OPENAI_API_KEY")


def call_openai_model(
    prompt,
    client,
    model_name="gpt-4o-mini",
    temperature=0.7,
    max_tokens=512,
    num_return_sequences=1,
):
    # price per 1M tokens in USD
    # https://platform.openai.com/docs/pricing
    MODEL_PRICES = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "o1-mini": {"input": 1.10, "output": 4.40},
        "o1": {"input": 15.00, "output": 60.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
    }
    
    try:
        if model_name == 'gpt-5':
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                n=num_return_sequences, 
            )
        else:
            
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
        
        return outputs, cost, total_toks
        
    except Exception as e:
        print(f"Error calling {model_name}: {e}")
        return [""], 0.0
    
    

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )    

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a HF model + tokenizer with sane defaults.
    """
    key = model_name.lower()
    if key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {list(SUPPORTED_MODELS)}")

    model_dir = WEIGHTS_ROOT / SUPPORTED_MODELS[key]
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model weights not found at {model_dir}. "
            f"Please place weights under {WEIGHTS_ROOT}."
        )
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    
    if 'qwen' in model_name:
        attn_implementation="flash_attention_2"
    else:
        attn_implementation="sdpa"  # default
        
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    model.eval()
    eos_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id
    if getattr(model, "config", None) is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None and gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = tokenizer.pad_token_id
    logging.info(f"Loaded model: {model_name} from {model_dir}")
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_name', type=str, default="causal_code")
    parser.add_argument('--data_split', type=str, default="train")
    parser.add_argument(
        "--data_model",
        type=str,
        default="qwen2.5-7b-instruct",
        choices=tuple(SUPPORTED_MODELS.keys()),
        help="Model used to generate data / CoT.",
    )
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    # call_gpt, store true
    parser.add_argument('--call_gpt', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    args.data_model = 'qwen2.5-7b-instruct'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model, tokenizer = load_model(args.data_model)
    