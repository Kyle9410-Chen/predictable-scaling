import torch
import wandb
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from predictable_scaling.scheduler import TokenLRScheduler

class LogConfig:
    def __init__(self, name, log_every: int=10,eval_every: int=100, tags: list[str]=["baseline"], config: dict=None):
        self.project = "lab6-predictable-scaling"
        self.name = name
        self.log_every = log_every
        self.eval_every = eval_every
        self.tags = tags
        self.config = config

def train(
        model: DistributedDataParallel, 
        opt,
        train_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
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
    last_time = time.perf_counter()
    last_token = 0
    tokens_seen = 0
    step = 0
    lr = 1e-3

    for step, batch in enumerate(train_loader, start=1):
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

            now = time.perf_counter()
            tokens_per_sec = (tokens_seen - last_token) / (now - last_time)
            last_time, last_token = time.perf_counter(), tokens_seen

            if dist_info.is_main:
                wandb.log(
                    {
                        "train/loss": avg_loss.item(), 
                        "train/loss_max": max_loss.item(), 
                        "train/lr": float(opt.param_groups[0]["lr"]),
                        "train/total_tokens_seen": tokens_seen,
                        "train/tokens_per_second": tokens_per_sec
                    }, 
                    step=step
                )
                print(f"step={step} tokens={tokens_seen:,} loss={avg_loss.item():.4f} loss_max={max_loss.item():.4f}")

        # evaluation
        if step % log_config.eval_every == 0:
            eval_start = time.perf_counter()
            eval_loss = eval(model, eval_loader, dist_info)
            eval_end = time.perf_counter()
            last_time = last_time + (eval_end - eval_start) # leave the eval time out of tokens/sec calculation
            if dist_info.is_main:
                wandb.log(
                    {
                        "eval/loss": eval_loss,
                        "eval/perplexity": torch.exp(torch.tensor(eval_loss)),
                        "eval/seconds": eval_end - eval_start
                    },
                    step=step
                )

        if tokens_seen >= max_tokens:
            break

@torch.no_grad()
def eval(model, eval_loader: torch.utils.data.DataLoader, dist_info):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=dist_info.device)

    for batch in eval_loader:
        batch = {k: v.to(dist_info.device, non_blocking=True) for k, v in batch.items()}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            ).loss
        n = (batch["labels"][:, 1:] != -100).sum()
        totals[0] += loss.double() * n
        totals[1] += n
    
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()

    return (totals[0] / totals[1]).item()