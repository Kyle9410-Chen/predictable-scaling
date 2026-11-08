from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import wandb
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from predictable_scaling.scheduler import TokenLRScheduler
from predictable_scaling.eval import eval
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from contextlib import nullcontext

if TYPE_CHECKING:
    from predictable_scaling.main import DistInfo

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
        dist_info: DistInfo, 
        max_tokens: int,
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

    # critical batch size logging
    ema_g2 = ema_trS = None

    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(dist_info.device, non_blocking=True) for k, v in batch.items()}

        measure = step % log_config.log_every == 0

        opt.zero_grad()

        # we need to take the batch loss in different device. so use no_sync
        with (model.no_sync() if measure else nullcontext()):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids = batch["input_ids"],
                    attention_mask = batch["attention_mask"],
                    labels=batch["labels"]
                ).loss
            loss.backward()

        step_tokens = (batch["labels"][:, 1:] != -100).sum()
        dist.all_reduce(step_tokens, op=dist.ReduceOp.SUM)
    
        # log grad norm and critical batchsize
        if step % log_config.log_every == 0:
            grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
            flat = _flatten_dense_tensors(grads)

            # McCandlish et al. 2018, An Empirical Model of Large-Batch Training.
            # |G_small|^2, averaged over ranks (each rank is one small-batch sample)
            small_batch_grad_square = flat.float().pow(2).sum()
            dist.all_reduce(small_batch_grad_square, op=dist.ReduceOp.AVG)

            # Sync the overall grad with all ranks
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(synced)
            big_batch_grad_square = flat.float().pow(2).sum()

            # get the batch size in tokens
            B_small: float = step_tokens.item() / dist_info.world_size
            B_big: float = step_tokens.item()

            # calculate gradient square and noise
            g2 = (B_big * big_batch_grad_square - B_small * small_batch_grad_square) / (B_big - B_small)
            trS = (small_batch_grad_square - big_batch_grad_square) / (1/B_small - 1/B_big)

            # average the noise estimator over time (EMA)
            ema_g2 = g2.item() if ema_g2 is None else 0.95 * ema_g2 + 0.05 * g2.item()
            ema_trS = trS.item() if ema_trS is None else 0.95 * ema_trS + 0.05 * trS.item()

            # critical batch size
            B_critical_tokens_raw = trS / g2
            B_critical_tokens = ema_trS / ema_g2

            total_norm = big_batch_grad_square.sqrt()

            if dist_info.is_main:
                wandb.log(
                    {
                        "train/grad_norm": total_norm.item(),
                        "train/B_critical_tokens_ema": B_critical_tokens,
                        "train/B_critical_tokens_raw": B_critical_tokens_raw
                    }, step=step)

        scheduler.update(tokens_seen)
        opt.step()

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