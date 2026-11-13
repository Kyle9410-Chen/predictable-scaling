import os
import torch
import datetime
import traceback
import argparse
import sys
import logging
import torch.distributed as dist

from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel

from predictable_scaling.optimizer import muon
from predictable_scaling.dataloader import load_dataset
from predictable_scaling.model import build_model, save_model
from predictable_scaling.train import train, LogConfig, BranchConfig, CkptConfig
from predictable_scaling.scheduler import TokenLRScheduler, wsd_shape
from predictable_scaling.checkpoint import load_model_weights, load_train_state

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
    p.add_argument("--max-tokens", type=float, default=30e9)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--name", type=str, default="training")
    p.add_argument("--tags", type=str, default="") # tag is comma seperated
    p.add_argument("--batch-size", type=int, default=4) # number of sequence per device per batch
    p.add_argument("--decay-frac", type=float, default=0.2)
    p.add_argument("--muon_lr", type=float, default=2.5e-3)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--adam_lr", type=float, default=1.2e-3)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--tailema-start", type=int) # the step to start tail ema
    p.add_argument("--tailema-tau", type=int)
    p.add_argument("--tailema-lambda", type=float)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--adam_eps", type=float, default=1e-8)
    p.add_argument("--scale", type=str, default="150M")
    p.add_argument("--save-dir", type=str, default="models")
    p.add_argument("--profile", action="store_true")

    # checkpoint and branch
    p.add_argument("--ckpt-at", type=str, default="") # comma seperated value. save checkpoint at token count
    p.add_argument("--branch-from", type=str, default="") # the checkpoint you want to branch from
    
    return p.parse_args()

def main():
    dist_info = setup_distributed()
    args = get_args()
    seq_len = 8192
    max_tokens = int(args.max_tokens)
    save_dir = f"{args.save_dir}/{args.name}"

    try:
        src = args.branch_from if args.branch_from != "" else None
        
        model = build_model(scale=args.scale).to(dist_info.device)
        if src:
            load_model_weights(src, model, dist_info.device)

        torch._dynamo.config.ignore_logging_functions.add(logging.Logger.warning_once)
        # raise the recompile limit, since the default 8 is less than the layers required by llama
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.accumulated_recompile_limit = 512
        for layer in model.model.layers:
            layer.compile()
        
        model = DistributedDataParallel(model, device_ids=[dist_info.local_rank])

        opt = muon(model, adam_lr=args.adam_lr, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2, muon_lr=args.muon_lr, muon_momentum=args.muon_momentum, wd=args.wd, adam_eps=args.adam_eps)
        start_state, skip, progress = None, None, None
        if src:
            start_state, progress = load_train_state(src, opt, dist_info)
            skip = {shard: frag + 1 for shard, frag in progress.items()} # skip partly read fragment

        train_loader = load_dataset(dist_info.rank, dist_info.world_size, split="train", batch_size=args.batch_size, seq_len=seq_len, skip=skip)
        eval_loader = load_dataset(dist_info.rank, dist_info.world_size, split="eval", workers=4)
        shape = wsd_shape(decay_frac=args.decay_frac)
        scheduler = TokenLRScheduler(opt, shape, max_tokens)

        run_config = {
            **vars(args)
        }

        branch_config = None
        if src:
            branch_config = BranchConfig(
                start_state=start_state,
                start_progress=progress
            )

        ckpt_config = None
        if args.ckpt_at != "":
            ckpt_at = sorted(int(float(x)) for x in args.ckpt_at.split(","))
            ckpt_config = CkptConfig(ckpt_at=ckpt_at, save_dir=save_dir)

        # guard tail ema setting has to be all set or all un set
        given = [args.tailema_start is not None, args.tailema_tau is not None, args.tailema_lambda is not None]
        if any(given) and not all(given):
            raise ValueError(f"Tail EMA args incomplete. Got --tailema-start={args.tailema_start}, --tailema-tau={args.tailema_tau}, --tailema-lambda={args.tailema_lambda}")

        train(
            model=model,
            opt=opt, 
            train_loader=train_loader, 
            eval_loader=eval_loader, 
            scheduler=scheduler,
            dist_info=dist_info, 
            max_tokens=max_tokens, 
            token_per_step=dist_info.world_size * (args.batch_size * seq_len - 1),
            tailema_start=args.tailema_start,
            tailema_tau=args.tailema_tau,
            tailema_lambda=args.tailema_lambda,
            log_config=LogConfig(
                name=args.name, 
                log_every=args.log_every,
                tags=args.tags.split(","), 
                eval_every=args.eval_every, 
                config=run_config,
                profile=args.profile
                ),
            branch_config=branch_config,
            ckpt_config=ckpt_config
            )

        dist.barrier()
        if dist_info.is_main:
            save_model(model, save_dir)
        dist.barrier()

    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()