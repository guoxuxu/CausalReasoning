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
    "qwen2.5-7b-instruct": "Qwen2.5-7B-Instruct",
}
WEIGHTS_ROOT = Path("/mnt/hdd/weights")


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
    parser.add_argument('--results_path', type=str, default="Results")
    parser.add_argument('--map_path', type=str, default="Map")
    parser.add_argument(
        "--answer_model",
        type=str,
        default="qwen2.5-7b-instruct",
        choices=tuple(SUPPORTED_MODELS.keys()),
        help="Model used to generate answers / CoT.",
    )
    parser.add_argument(
        "--probe_model",
        type=str,
        default="qwen2.5-7b-instruct",
        choices=tuple(SUPPORTED_MODELS.keys()),
        help="Model used to generate causal maps.",
    )
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--num_return_sequences', type=int, default=1)
    # call_gpt, store true
    parser.add_argument('--call_gpt', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--generate_map', action='store_true')
    parser.add_argument('--integrate_map', action='store_true')
    parser.add_argument('--use_map', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    args.answer_model = 'qwen2.5-7b-instruct'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    model, tokenizer = load_model(args.answer_model)
    