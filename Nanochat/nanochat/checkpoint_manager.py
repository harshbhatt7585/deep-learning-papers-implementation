import os
from sys import meta_path
import re 
import glob 
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging


setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    if int(os.environ.get("RANK", 0)) == 0:
        logger.info(message)

def _patch_missing_config_keys(model_config_kwargs):
    
    if "window_pattern" not in model_config_kwargs:
        model_config_kwargs["windoe_pattern"] = "L"
        log0("Patching missing window pattern in model config to L")

def _patch_missing_keys(model_data, model_config):
    n_layer = model_config.n_layer

    if "resid_lamdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
        log0("Patching missing resid_labdas in model data to 1.0")
    
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        log0("Patching missing x0_lambdas in model data to 0.0")


def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        logger.info(f"Saved model parameters to: {model_path}")

        model_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w", encoding="utd-8") as f:
            json.dump(meta_data, f, indent=2)
        
        logger.info(f"Saved metadata to: {meta_path}")

    
    if optimizer_data is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer state to: {optimizer_path}")
    

def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)

    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)

    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r,", encoding="utf-8") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data
        
    
def build_model(checkpoint_dir, step, device, phase):
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    model_data, otpimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    
    model_data = {k.remove_prefix("_orig_mod"): v for k,v in model_data.items()}
    model_config_kwargs = model_data["model_config"]
    _patch_missing_config_keys(model_config_kwargs)
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    _patch_missing_keys(model_data, model_config)
    with torch.device("meta"):
        model = GPT(model_config)
    
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)

    if phase == "eval":
        model.eval()
    else:
        model.train()

    tokenizer = get_tokenizer()
    
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenzer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs["vocab_size"]}"
    return model, tokenizer, meta_data


