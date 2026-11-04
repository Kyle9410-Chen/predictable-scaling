import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb

class LogConfig:
    def __init__(self, name, log_every: int=10, tags: list[str]=["baseline"], config: dict=None):
        self.project = "lab6-predictable-scaling"
        self.name = name
        self.log_every = log_every
        self.tags = tags
        self.config = config

def train(model: DistributedDataParallel, opt, loader, info, max_tokens, log_config: LogConfig=None):
    with wandb.init(
        project="lab6-predictable-scaling",
        name=log_config.name,
        tags=log_config.tags,
        config=log_config.config,
    ) as run:
        model.train()
        tokens_seen = 0
        step = 0
        lr = 1e-3

        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(info.device, non_blocking=True) for k, v in batch.items()}

            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                        input_ids = batch["input_ids"],
                        attention_mask = batch["attention_mask"],
                        labels=batch["labels"]
                    ).loss
            loss.backward()
            opt.step()

            step_tokens = (batch["labels"][:, 1:] != -100).sum()
            dist.all_reduce(step_tokens, op=dist.ReduceOp.SUM)
            tokens_seen += step_tokens.item()

            if step % log_config.log_every == 0:
                avg_loss = loss.detach().clone()
                max_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(max_loss, op=dist.ReduceOp.MAX)
                if info.is_main:
                    run.log(
                        {
                            "loss": avg_loss.item(), 
                            "loss_max": max_loss.item(), 
                            "lr": lr,
                            "token_seen": tokens_seen
                        }, 
                        step=step
                    )
                    print(f"step={step} tokens={tokens_seen:,} loss={avg_loss.item():.4f} loss_max={max_loss.item():.4f}")

            if tokens_seen >= max_tokens:
                break