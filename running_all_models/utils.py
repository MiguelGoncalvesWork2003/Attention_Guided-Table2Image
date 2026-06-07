"""
Reproducibility utilities for the benchmark experiments.
Provides seed setting and statistical aggregation (mean, std, 95% CI).
"""
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mean_std_ci(scores, confidence=0.95):
    arr = np.array(scores)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    n = len(arr)
    stderr = std / np.sqrt(n)
    ci = 1.96 * stderr       # 95% normal approximation
    return mean, std, ci