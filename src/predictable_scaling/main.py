import os
import torch
import datetime
import traceback
import argparse
import sys
import torch.distributed as dist

from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel

from predictable_scaling.optimizer import muon
from predictable_scaling.dataloader import load_dataset
from predictable_scaling.model import build_model, save_model
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
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=5))
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
    p.add_argument("--batch-size", type=int, default=4) # number of sequence per device per batch
    p.add_argument("--adam_lr", type=float, default=1e-3)
    p.add_argument("--muon_lr", type=float, default=1e-2)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--wd", type=float, default=0.1)
    
    return p.parse_args()

def main():
    dist_info = setup_distributed()
    args = get_args()
    seq_len = 8192

    try:
        model = build_model().to(dist_info.device)
        opt = muon(model, adam_lr=args.adam_lr, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, muon_lr=args.muon_lr, wd=args.wd)
        train_loader = load_dataset(dist_info.rank, dist_info.world_size, split="train", batch_size=args.batch_size, seq_len=seq_len)
        eval_loader = load_dataset(dist_info.rank, dist_info.world_size, split="eval", workers=4)
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank])
        shape = wsd_shape()
        scheduler = TokenLRScheduler(opt, shape, args.max_tokens)

        run_config = {
            **vars(args)
        }

        train(
            model=model,
            opt=opt, 
            train_loader=train_loader, 
            eval_loader=eval_loader, 
            scheduler=scheduler,
            dist_info=dist_info, 
            max_tokens=args.max_tokens, 
            token_per_step=dist_info.world_size * (args.batch_size * seq_len - 1),
            log_config=LogConfig(name=args.name, log_every=args.log_every, tags=args.tags.split(","), eval_every=500, config=run_config)
            )

        dist.barrier()
        if dist_info.is_main:
            save_model(model, f"models/{args.name}")
        dist.barrier()

    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()