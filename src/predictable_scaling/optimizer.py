from transformers import AutoModelForCausalLM
from muonium import Muon

def muon(model: AutoModelForCausalLM, adam_lr=1e-3, adam_beta1=0.9,adam_beta2=0.95, muon_lr=1e-2, wd=0.1) -> Muon:
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    optimizer = Muon(
        [
            {"params": muon_params, "lr": muon_lr},
            {"params": adam_params, "algorithm": "adamw", "lr": adam_lr, "betas": (adam_beta1, adam_beta2)},
        ],
        wd=wd
    )

    return optimizer