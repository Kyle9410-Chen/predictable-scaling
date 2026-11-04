from transformers import AutoModelForCausalLM
from muonium import Muon

def muon(model: AutoModelForCausalLM) -> Muon:
    muon_params = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed" not in n]
    adam_params = [p for n, p in model.named_parameters() if p.ndim < 2 or "embed" in n]

    optimizer = Muon(
        [
            {"params": muon_params},
            {"params": adam_params, "algorithm": "adamw"},
        ]
    )

    return optimizer