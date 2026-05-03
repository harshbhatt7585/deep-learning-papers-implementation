import os
from sys import meta_path
import reimport glob 
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
        