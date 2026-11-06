import os
import torch
import argparse
import torch.distributed as dist

from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel

from predictable_scaling.optimizer import muon
from predictable_scaling.dataloader import load_dataset
from predictable_scaling.model import build_model
from predictable_scaling.train import train
from predictable_scaling.scheduler import TokenLRScheduler, wsd_shape
from predictable_scaling.train import LogConfig

@dataclass(frozen=True)
class DistInfo:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

def setup_distributed() -> DistInfo:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return DistInfo(
        rank = dist.get_rank(),
        world_size = dist.get_world_size(),
        local_rank = local_rank,
        device = torch.device("cuda", local_rank)
    )

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-tokens", type=int, default=30e9)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--name", type=str, default="training")
    p.add_argument("--tags", type=str, default="") # tag is comma seperated

    return p.parse_args()

def main():
    dist_info = setup_distributed()
    args = get_args()

    try:
        model = build_model().to(dist_info.device)
        opt = muon(model)
        train_loader = load_dataset(dist_info.rank, dist_info.world_size, split="train")
        eval_loader = load_dataset(dist_info.rank, dist_info.world_size, split="eval", workers=4)
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank])
        shape = wsd_shape()
        scheduler = TokenLRScheduler(opt, shape, args.max_tokens)

        train(
            model=model,
            opt=opt, 
            train_loader=train_loader, 
            eval_loader=eval_loader, 
            scheduler=scheduler,
            dist_info=dist_info, 
            max_tokens=args.max_tokens, 
            log_config=LogConfig(name=args.name, log_every=args.log_every, tags=args.tags.split(","), eval_every=20)
            )

    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()