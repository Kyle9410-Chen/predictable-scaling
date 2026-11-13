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
        if decay_frac == 0 or p < decay_start:
            return 1.0
        return 1 - (1 - min_ratio) * (p - decay_start) / decay_frac
    
    return shape

class TokenLRScheduler:
    def __init__(self, opt, max_tokens: int, shape = None, shapes: list = []): # shapes will be applied according to the order of param groups
        self.opt = opt
        self.base_lrs = [float(g["lr"]) for g in opt.param_groups]
        self.max_tokens = max_tokens

        self.shapes = []
        if shapes:
            if len(shapes) != len(opt.param_groups):
                raise ValueError(f"got {len(shapes)} shapes for {len(opt.param_groups)} param groups")
            self.shapes = list(shapes)
        elif shape is not None:
            self.shapes = [shape] * len(opt.param_groups)
        else:
            raise ValueError("No shape provided")

    def update(self, tokens_seen: int) -> None:
        p = min(tokens_seen / self.max_tokens, 1.0)
        for group, base, shape in zip(self.opt.param_groups, self.base_lrs, self.shapes):
            lr = base * shape(p)
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(lr)
            else:
                group["lr"] = lr