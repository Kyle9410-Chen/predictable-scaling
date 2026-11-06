import torch
import wandb
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from predictable_scaling.scheduler import TokenLRScheduler

class LogConfig:
    def __init__(self, name, log_every: int=10, tags: list[str]=["baseline"], config: dict=None):
        self.project = "lab6-predictable-scaling"
        self.name = name
        self.log_every = log_every
        self.tags = tags
        self.config = config

def train(
        model: DistributedDataParallel, 
        opt,
        loader,
        scheduler: TokenLRScheduler,
        dist_info, 
        max_tokens,
        log_config: LogConfig=None
        ):
    
    if dist_info.is_main:
        wandb.init(
            project="lab6-predictable-scaling",
            name=log_config.name,
            tags=log_config.tags,
            config=log_config.config,
            )
    
    model.train()
    tokens_seen = 0
    step = 0
    lr = 1e-3

    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(dist_info.device, non_blocking=True) for k, v in batch.items()}

        opt.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels=batch["labels"]
                ).loss
        loss.backward()
        
        # log grad norm
        if step % log_config.log_every == 0:
            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]),
                2
            )
            if dist_info.is_main:
                wandb.log({"train/grad_norm": total_norm.item()}, step=step)

        scheduler.update(tokens_seen)
        opt.step()

        step_tokens = (batch["labels"][:, 1:] != -100).sum()
        dist.all_reduce(step_tokens, op=dist.ReduceOp.SUM)
        tokens_seen += step_tokens.item()

        # log loss, token_seen, lr
        if step % log_config.log_every == 0:
            avg_loss = loss.detach().clone()
            max_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)
            if dist_info.is_main:
                wandb.log(
                    {
                        "train/loss": avg_loss.item(), 
                        "train/loss_max": max_loss.item(), 
                        "train/lr": float(opt.param_groups[0]["lr"]),
                        "train/total_tokens_seen": tokens_seen
                    }, 
                    step=step
                )
                print(f"step={step} tokens={tokens_seen:,} loss={avg_loss.item():.4f} loss_max={max_loss.item():.4f}")

        if tokens_seen >= max_tokens:
            break