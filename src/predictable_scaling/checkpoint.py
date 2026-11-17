"""Checkpoint save/load for WSD branching.

On-disk layout
--------------
A run's directory is ``{--save-dir}/{--name}``. It holds both the final
HF model (written by ``save_model`` at the end) and the checkpoints::

    {save_dir}/{name}/
    ├── config.json, model.safetensors, tokenizer*.json  # final model (save_model)
    ├── tok_<tokens_seen>/                                # one checkpoint
    │   ├── model.pt          # model.module.state_dict(); written by rank 0
    │   ├── train_state.pt    # JSON (despite the .pt suffix); written by rank 0:
    │   │                     #   step, tokens_seen, ema_g2, ema_trS, wandb_id, world_size
    │   ├── rank<r>.pt        # one per rank r = 0..world_size-1:
    │   │                     #   {"opt": opt.state_dict(),   # this rank's DistributedMuon shard
    │   │                     #    "opt_by_name": {param_name: state},  # same state, keyed by name
    │   │                     #    "data": {shard: frag}}     # this rank's workers' data position
    │   └── KEEP              # empty marker: a --ckpt-at checkpoint; never pruned
    └── tok_<tokens_seen>.tmp/                            # in-progress save; renamed when complete

Naming
------
- ``tok_<N>``: N is the exact ``tokens_seen`` when the checkpoint was saved. Saves
  happen on step boundaries, so N is slightly above the ``--ckpt-at`` target
  (e.g. ``--ckpt-at 8e9`` gives ``tok_8000256000``). Use ``ls`` to get the real name.
- ``.tmp`` suffix: a save that hasn't finished, or one that crashed partway.
  Ignored by ``latest_checkpoint`` and ``prune_checkpoints``; safe to delete.
- ``KEEP``: present only on ``--ckpt-at`` checkpoints. Rolling checkpoints
  (``ckpt_every``) have no marker and are pruned to the last ``keep_last``.

Restore requirements
--------------------
- Same world size (asserted on load): DistributedMuon's owner map and the
  per-rank optimizer files depend on it.
- Same number of DataLoader workers per rank: shard ids in ``"data"`` are
  ``rank * num_workers + worker_id``.
- Same model scale.
- Optimizer *state* comes from the checkpoint; *hyperparameters* (lr, betas,
  eps, momentum, wd) come from the current command-line args / --hp-config.
- For DistributedMuon, state is matched by parameter name, so a branch may use a
  different HP grouping than its trunk. Checkpoints without "opt_by_name" are
  read with the legacy 2-group layout.
"""

import shutil
import os
import glob
import json
import torch

import torch.distributed as dist

def save_checkpoint(path: str, model: torch.nn.Module, opt, train_state: dict, data_progress: dict, dist_info):
    """All ranks call this. Writes <path>.tmp, then renames it to <path>."""
    tmp = path + ".tmp"
    if dist_info.is_main:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
    dist.barrier()

    if dist_info.is_main:
        torch.save(model.module.state_dict(), f"{tmp}/model.pt")
        with open(f"{tmp}/train_state.json", "w") as f:
            json.dump({**train_state, "world_size": dist_info.world_size}, f)

    # every rank
    payload = {"opt": opt.state_dict(), "data": data_progress}
    if _is_distributed_muon(opt):
        payload["opt_by_name"] = opt_state_by_name(opt, model)
    torch.save(payload, f"{tmp}/rank{dist_info.rank}.pt")
    dist.barrier()

    if dist_info.is_main:
        shutil.rmtree(path, ignore_errors=True)
        os.rename(tmp, path)
    dist.barrier()

def load_model_weights(path: str, model: torch.nn.Module, device):
    "Bare model before compile and DDP"
    model.load_state_dict(torch.load(f"{path}/model.pt", map_location=device))

def _is_distributed_muon(opt) -> bool:
    from predictable_scaling.optimizer import DistributedMuon
    return isinstance(opt, DistributedMuon)

def opt_state_by_name(opt, model) -> dict:
    "{param_name: state} for the params this rank holds optimizer state for"
    pname = {p: n for n, p in model.module.named_parameters()}
    return {pname[p]: s for p, s in opt.state.items() if len(s) > 0}

def legacy_state_by_name(sd: dict, model, rank: int, world: int) -> dict:
    """Name the states of a checkpoint saved by the old 2-group DistributedMuon
    ([this rank's muon matrices, all adam params]); index order = that group order."""
    named = list(model.module.named_parameters())
    muon = [(n, p) for n, p in named if p.ndim >= 2 and "embed" not in n]
    adam = [n for n, p in named if p.ndim < 2 or "embed" in n]
    order = sorted(range(len(muon)), key=lambda i: -muon[i][1].numel())
    owner = {muon[i][0]: k % world for k, i in enumerate(order)}
    names = [n for n, _ in muon if owner[n] == rank] + adam

    n_saved = sum(len(g["params"]) for g in sd["param_groups"])
    if len(sd["param_groups"]) != 2 or n_saved != len(names):
        raise ValueError(f"checkpoint optimizer has {len(sd['param_groups'])} groups / {n_saved} params, "
                         f"expected the legacy 2-group layout with {len(names)} params")
    return {names[int(i)]: s for i, s in sd["state"].items()}

STATE_KEY = {"muon": "momentum_buffer", "adamw": "exp_avg"}

def load_opt_by_name(opt, model, by_name: dict):
    """Load state keyed by param name into the current group layout; HPs stay current."""
    pname = {p: n for n, p in model.module.named_parameters()}
    sd = opt.state_dict()  # current layout and current HPs
    state, missing, idx = {}, [], 0
    for g in opt.param_groups:
        key = STATE_KEY.get(g["algorithm"])
        for p in g["params"]:
            n = pname[p]
            if n not in by_name:
                missing.append(n)
            elif key is not None and key not in by_name[n]:
                raise ValueError(f"{n}: saved state {sorted(by_name[n])} does not fit "
                                 f"algorithm {g['algorithm']} (group {g.get('name')})")
            else:
                state[idx] = by_name[n]
            idx += 1
    if missing:
        raise ValueError(f"no saved optimizer state for {len(missing)} params on this rank, "
                         f"e.g. {missing[:3]} (did muon ownership change?)")

    current = [dict(g) for g in opt.param_groups]
    sd["state"] = state
    opt.load_state_dict(sd)  # torch casts state to each param's device/dtype
    for group, cur in zip(opt.param_groups, current):  # keep the exact current group objects (lr tensors)
        group.update(cur)

def load_train_state(path: str, opt, dist_info, model=None):
    "Optimizer states, data progress"

    with open(f"{path}/train_state.json") as f:
        train_state = json.load(f)
    if train_state["world_size"] != dist_info.world_size:
        raise ValueError(f"checkpoint world_size={train_state['world_size']}, "
                         f"current={dist_info.world_size}; DistributedMuon owner map differs")

    raw = torch.load(f"{path}/rank{dist_info.rank}.pt", map_location="cpu", weights_only=False)

    # restore optimizer state, but not hyperparams
    if _is_distributed_muon(opt):
        # by parameter name: works across any regrouping of params into HP groups
        if model is None:
            raise ValueError("load_train_state needs model for DistributedMuon")
        by_name = raw.get("opt_by_name")
        if by_name is None:
            by_name = legacy_state_by_name(raw["opt"], model, dist_info.rank, dist_info.world_size)
        load_opt_by_name(opt, model, by_name)
    else:
        current = [{key: value for key, value in group.items()} for group in opt.param_groups] # current optimizer setting
        opt.load_state_dict(raw["opt"]) # restore both state and HP
        for group, cur in zip(opt.param_groups, current): # keep current HP
            group.update(cur)

    # restore data progress
    data_progress = {int(key): value for key, value in raw["data"].items()}

    return train_state, data_progress

# checkpoint management helpers
def latest_checkpoint(path: str) -> str | None:
    ckpts = [p for p in glob.glob(f"{path}/tok_*") if not p.endswith(".tmp")]
    ckpts.sort(key=lambda p: int(p.rsplit("_", 1)[-1]))
    return ckpts[-1] if ckpts else None

def prune_checkpoints(path: str, keep_last: int):
    "rank 0 only. delete old rolling checkpoints"
    rolling = [p for p in glob.glob(f"{path}/tok_*") if not p.endswith(".tmp") and not os.path.exists(f"{p}/KEEP")]
    rolling.sort(key=lambda p: int(p.rsplit("_", 1)[-1]))
    for old in rolling[:-keep_last] if keep_last > 0 else rolling:
        shutil.rmtree(old, ignore_errors=True)