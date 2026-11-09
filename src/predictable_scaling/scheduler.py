import math
from collections.abc import Callable
import torch

def cosine_shape(warmup_frac=0.0, min_ratio=0.0) -> Callable[[float], float]:
    def shape(p: float) -> float:
        if p < warmup_frac:
            return p / warmup_frac
        t = (p - warmup_frac) / (1 - warmup_frac)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
    
    return shape

def wsd_shape(warmup_frac=0.01, decay_frac=0.2, min_ratio=0.0) -> Callable[[float], float]:
    decay_start = 1 - decay_frac

    def shape(p: float) -> float:
        if p < warmup_frac:
            return p / warmup_frac
        if p < decay_start:
            return 1.0
        return 1 - (1 - min_ratio) * (p - decay_start) / decay_frac
    
    return shape

class TokenLRScheduler:
    def __init__(self, opt, shape, max_tokens: int):
        self.opt = opt
        self.shape = shape
        self.max_tokens = max_tokens
        self.base_lrs = [float(g["lr"]) for g in opt.param_groups]
    

    def update(self, tokens_seen: int) -> None:
        ratio = self.shape(min(tokens_seen / self.max_tokens, 1.0))
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(base * ratio)
            else:
                group["lr"] = base * ratio