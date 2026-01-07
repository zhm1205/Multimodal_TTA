"""
Utility functions and helper classes
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Union, Literal


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics
        
        Args:
            val: New value
            n: Number of samples for the new value
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_random_seed(seed: int, mode: Literal["off", "practical", "strict"] = "practical") -> None:
    """
    Reproducibility presets:
      - "off":       速度优先，允许非确定性
      - "practical": 实用可复现（推荐），不调用 use_deterministic_algorithms
      - "strict":    严格确定性，需要 CUBLAS_WORKSPACE_CONFIG，可能更慢
    """
    import os, random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if mode == "off":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return

    if mode == "practical":
        # practical reproducibility, common and stable
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return

    if mode == "strict":
        # strict reproducibility: requires setting this environment variable early in the process (the earlier the better)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        return

    raise ValueError("mode must be 'off' | 'practical' | 'strict'")