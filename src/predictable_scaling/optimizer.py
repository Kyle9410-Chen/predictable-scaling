from transformers import AutoModelForCausalLM
from muonium import Muon

def muon(model: AutoModelForCausalLM, adam_lr=1e-3, muon_lr=1e-2, wd=0.1) -> Muon:
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    optimizer = Muon(
        [
            {"params": muon_params, "lr": muon_lr},
            {"params": adam_params, "algorithm": "adamw", "lr": adam_lr},
        ],
        wd=wd
    )

    return optimizer