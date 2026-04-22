from typing import Optional
from decoder import ImageCaptioner
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from textwrap import fill

import os
import argparse
import yaml
import random
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | [%(levelname)s] | %(message)s",
    handlers=[
        logging.StreamHandler(),                 # print to console
        logging.FileHandler("logs/generate.log", mode='w')    # save to file
    ]
)
logger = logging.getLogger()


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config(config_path):
    """Load config file"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(BASE_DIR, 'config', 'main.yml')
    
    if not os.path.exists(config_path):
        config_path = os.path.join(BASE_DIR, config_path)
    
    if not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}, using default")
        config_path = default_config
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


class DefaultValues:
    """Default values for caption generation"""
    RESULT_DIR = "results"
    SAMPLE_SIZE = 20  # Number of validation images to sample
    MAX_TOKENS = 25
    TEMPERATURE = 0.7
    TOP_K = 0
    TOP_P = 0.9
    CONFIG_PATH = "config/main.yml"



@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_k: Optional[int] = None, 
                      
                     top_p: Optional[float] = None):
    """Sample next token from logits"""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        vals, idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        probs2 = torch.zeros_like(probs).scatter_(-1, idx, vals)
        probs = probs2 / probs2.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum > top_p
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        
        next_in_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_in_sorted)
        return next_token

    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_caption(model: ImageCaptioner,
                    image_features: torch.Tensor,
                    max_new_tokens: int = 25,
                    bos_id: int = 1,
                    eos_id: Optional[int] = None,
                    unk_id: Optional[int] = None,
                    handle_unk: bool = False,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None,
                    top_p: Optional[float] = None,
                    greedy: bool = False):
    """Generate caption from image features autoregressively"""
    model.eval()
    
    batch_size = image_features.size(0)
    device = image_features.device
    
    out = torch.full(
        (batch_size, 1),
        bos_id,
        dtype=torch.long,
        device=device
    )
    
    for step in range(max_new_tokens):
        logits, _, _ = model(image_features, out)
        last_logits = logits[:, -1, :]
        
        if handle_unk and not greedy and unk_id is not None:
            last_logits[:, unk_id] = -float('inf')
        
        if greedy:
            nxt = last_logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = sample_next_token(
                last_logits, 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
        
        out = torch.cat([out, nxt], dim=1)
        
        if eos_id is not None and (nxt.squeeze(-1) == eos_id).all():
            break
    
    return out

