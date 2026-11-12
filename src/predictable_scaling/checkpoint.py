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
    torch.save({"opt": opt.state_dict(), "data": data_progress}, f"{tmp}/rank{dist_info.rank}.pt")
    dist.barrier()

    if dist_info.is_main:
        shutil.rmtree(path, ignore_errors=True)
        os.rename(tmp, path)
    dist.barrier()

def load_model_weights(path: str, model: torch.nn.Module, device):
    "Bare model before compile and DDP"
    model.load_state_dict(torch.load(f"{path}/model.pt", map_location=device))

def load_train_state(path: str, opt, dist_info):
    "Optimizer states, data progress"

    with open(f"{path}/train_state.json") as f:
        train_state = json.load(f)
    if train_state["world_size"] != dist_info.world_size:
        raise ValueError(f"checkpoint world_size={train_state['world_size']}, "
                         f"current={dist_info.world_size}; DistributedMuon owner map differs")

    raw = torch.load(f"{path}/rank{dist_info.rank}.pt", map_location="cpu", weights_only=False)

    # restore optimizer state, but not hyperparams
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