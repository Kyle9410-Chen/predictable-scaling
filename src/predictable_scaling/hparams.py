"""Declarative per-module hyperparameters.

The YAML holds final values only; nothing is derived here. Layout::

    muon:  {momentum: 0.95, wd: 0.1}                       # defaults for every muon group
    adamw: {betas: [0.9, 0.999], eps: 1.0e-8, wd: 0.1}     # defaults for every adamw group
    groups:                                                # first regex match wins (top to bottom)
      attn_qo: {match: 'self_attn\\.(q|o)_proj', algorithm: muon, lr: 2.5e-3}
      ...

A group key overrides the algorithm default. ``one_minus_betas: [b1, b2]`` may be
given instead of ``betas`` so the 1-beta columns of the transfer sheet can be pasted.
An optional ``soap: {shampoo_beta: 0.95}`` section (or ``one_minus_shampoo_beta``) holds the
SOAP hyperparameters; it is only used with ``--opt soap-muon``.
CLI overrides: ``--hp groups.embed.lr=3e-3 --hp adamw.wd=0.05``.
"""

import re
import yaml

ALLOWED = {
    "muon": {"lr", "momentum", "wd", "nesterov"},
    "adamw": {"lr", "betas", "one_minus_betas", "eps", "wd"},
}
REQUIRED = {
    "muon": {"lr", "momentum", "wd"},
    "adamw": {"lr", "betas", "eps", "wd"},
}


def _num(v):
    # PyYAML parses "1e-8" (no dot) as a string; accept it as a float anyway
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    if isinstance(v, list):
        return [_num(x) for x in v]
    return v


def load_hp_config(path: str, overrides: list[str] | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--hp expects key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        *path_, last = key.split(".")
        d = cfg
        for k in path_:
            d = d.setdefault(k, {})
        d[last] = _num(yaml.safe_load(val))
    return cfg


def resolve_groups(cfg: dict, model) -> list[dict]:
    """Assign every parameter of the bare model to one group and merge its HPs.

    Returns [{"name", "algorithm", "param_names", <hp>...}] in config order.
    """
    groups = cfg["groups"]
    pats = {g: re.compile(spec["match"]) for g, spec in groups.items()}

    assign = {g: [] for g in groups}
    for n, _ in model.named_parameters():  # tied lm_head is deduplicated here
        g = next((g for g in groups if pats[g].search(n)), None)
        if g is None:
            raise ValueError(f"parameter {n} matches no HP group")
        assign[g].append(n)

    specs = []
    for g, spec in groups.items():
        algo = spec.get("algorithm")
        if algo not in ALLOWED:
            raise ValueError(f"group {g}: algorithm must be one of {sorted(ALLOWED)}, got {algo!r}")
        if not assign[g]:
            raise ValueError(f"group {g} matched no parameters")

        defaults = cfg.get(algo, {}) or {}
        own = {k: v for k, v in spec.items() if k not in ("match", "algorithm")}
        for src, d in (("defaults", defaults), (g, own)):
            bad = set(d) - ALLOWED[algo]
            if bad:
                raise ValueError(f"{algo} {src}: unknown keys {sorted(bad)}")
        if "betas" in own and "one_minus_betas" in own:
            raise ValueError(f"group {g}: set betas or one_minus_betas, not both")

        hp = {k: _num(v) for k, v in {**defaults, **own}.items()}
        # a group-level betas/one_minus_betas replaces the default of the other form
        if "betas" in own:
            hp.pop("one_minus_betas", None)
        elif "one_minus_betas" in own:
            hp.pop("betas", None)
        if "one_minus_betas" in hp:
            if "betas" in hp:
                raise ValueError(f"{algo} defaults: set betas or one_minus_betas, not both")
            hp["betas"] = [1.0 - b for b in hp.pop("one_minus_betas")]
        if "betas" in hp:
            hp["betas"] = tuple(float(b) for b in hp["betas"])

        missing = REQUIRED[algo] - set(hp)
        if missing:
            raise ValueError(f"group {g}: missing {sorted(missing)}")
        for b in (*hp.get("betas", ()), hp.get("momentum", 0.5)):
            if not 0.0 <= b < 1.0:
                raise ValueError(f"group {g}: beta/momentum {b} outside [0, 1) "
                                 "(did you paste 1-beta into betas?)")

        specs.append({"name": g, "algorithm": algo, "param_names": assign[g], **hp})
    return specs


SOAP_KEYS = {"shampoo_beta", "one_minus_shampoo_beta", "precondition_frequency", "max_precond_dim"}

def resolve_soap(cfg: dict) -> dict:
    """SOAP kwargs from the optional ``soap`` section; missing keys keep SOAP's defaults."""
    hp = {k: _num(v) for k, v in (cfg.get("soap") or {}).items()}
    bad = set(hp) - SOAP_KEYS
    if bad:
        raise ValueError(f"soap: unknown keys {sorted(bad)}")
    if "one_minus_shampoo_beta" in hp:
        if "shampoo_beta" in hp:
            raise ValueError("soap: set shampoo_beta or one_minus_shampoo_beta, not both")
        hp["shampoo_beta"] = 1.0 - hp.pop("one_minus_shampoo_beta")
    if not 0.0 <= hp.get("shampoo_beta", 0.95) < 1.0:
        raise ValueError(f"soap: shampoo_beta {hp['shampoo_beta']} outside [0, 1)")
    return hp


def describe(specs: list[dict]) -> dict:
    """Resolved HPs without parameter lists, for wandb config / printing."""
    return {s["name"]: {**{k: v for k, v in s.items() if k not in ("name", "param_names")},
                        "n_params": len(s["param_names"])}
            for s in specs}
