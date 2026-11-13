from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import wandb
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from predictable_scaling.scheduler import TokenLRScheduler
from predictable_scaling.eval import eval
from predictable_scaling.checkpoint import save_checkpoint, prune_checkpoints
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from contextlib import nullcontext
from itertools import batched

if TYPE_CHECKING:
    from predictable_scaling.main import DistInfo

class LogConfig:
    def __init__(self, name, log_every: int=10,eval_every: int=100, tags: list[str]=["baseline"], config: dict=None, profile=False):
        self.project = "lab6-predictable-scaling"
        self.name = name
        self.log_every = log_every
        self.eval_every = eval_every
        self.tags = tags
        self.config = config
        self.profile = profile

class CkptConfig:
    def __init__(self, ckpt_every:int = 0, keep_last: int =2, ckpt_at: tuple[int, ...]=(), save_dir: str=""):
        self.ckpt_every = ckpt_every
        self.keep_last = keep_last
        self.ckpt_at = ckpt_at
        self.save_dir = save_dir

class BranchConfig:
    def __init__(self, start_state: dict=None, start_progress: dict=None):
        self.start_state = start_state
        self.start_progress = start_progress

def _t():
    torch.cuda.synchronize()
    return time.perf_counter()

def train(
        model: DistributedDataParallel, 
        opt,
        train_loader: torch.utils.data.DataLoader,
        eval_loader: torch.utils.data.DataLoader,
        scheduler: TokenLRScheduler,
        dist_info: DistInfo, 
        token_per_step: int,
        max_tokens: int,
        batch_accumulation: int=1,
        log_config: LogConfig=None,
        ckpt_config: CkptConfig=None,
        branch_config: BranchConfig=None,
    ):
    
    start_state = branch_config.start_state if branch_config else {}

    if dist_info.is_main:
        wandb.init(
            project="lab6-predictable-scaling",
            name=log_config.name,
            tags=log_config.tags,
            config=log_config.config,
            )
    
    model.train()

    # start from resume state if any
    step0 = start_state.get("step", 0)
    tokens_seen = start_state.get("tokens_seen", 0)

    # critical batch size logging
    ema_g2 = start_state.get("ema_g2")
    ema_trS = start_state.get("ema_trS")
    
    last_time = time.perf_counter()
    last_token = tokens_seen

    progress: dict[int, int] = dict(branch_config.start_progress) if branch_config else {}
    if ckpt_config is not None and ckpt_config.ckpt_every > 0:
        next_roll = (tokens_seen // ckpt_config.ckpt_every + 1) * ckpt_config.ckpt_every
    else:
        next_roll = None
    pending_at = [t for t in ckpt_config.ckpt_at if t > tokens_seen] if ckpt_config else []

    if log_config.profile:
        from collections import defaultdict
        timing = defaultdict(float)
        n_timed = 0
        t_end = _t()

    for step, micro_batch in enumerate(batched(train_loader, batch_accumulation), start=step0 + 1):
        if log_config.profile:
            t_data = _t()

        # update progress for this shard
        shard, frag = micro_batch[-1]["data_pos"]
        progress[shard] = frag

        measure = step % log_config.log_every == 0
        opt.zero_grad()
        loss_sum = 0.0

        if log_config.profile:
            t0 = _t()

        for i, batch in enumerate(micro_batch):
            batch.pop("data_pos")
            batch = {k: v.to(dist_info.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            torch._dynamo.mark_dynamic(batch["cu_seq_lens_q"], 0) # it change every batch. tell torch compile it is dynamic
            torch._dynamo.mark_dynamic(batch["cu_seq_lens_k"], 0)

            last = i == len(micro_batch) - 1
            sync = last and not measure # measure steps do the all reduce

            with (nullcontext() if sync else model.no_sync()):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss / batch_accumulation
                if log_config.profile:
                    t1 = _t()
                loss.backward()
            loss_sum += loss.detach()

        loss = loss_sum # mean loss over the full batch, for logging

        if log_config.profile:
            t2 = _t()

        # log grad norm and critical batchsize
        if measure:
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
            B_small: float = token_per_step / dist_info.world_size
            B_big: float = token_per_step

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

        if log_config.profile:
            t3 = _t()

        scheduler.update(tokens_seen)
        opt.step()

        if log_config.profile:
            t4 = _t()

        if log_config.profile:
            if step > 50 and not measure:
                timing["data"] += t_data - t_end
                timing["h2d"]  += t0 - t_data
                timing["fwd"]  += t1 - t0
                timing["bwd"]  += t2 - t1
                timing["opt"]  += t4 - t3
                n_timed += 1
            if measure and step > 50:
                timing["cbs"] += t3 - t2          # cost of one CBS measurement
                timing["n_cbs"] += 1

            if dist_info.is_main and n_timed and step % 50 == 0:
                ms = {k: 1e3 * v / n_timed for k, v in timing.items() if k not in ("cbs", "n_cbs")}
                cbs = 1e3 * timing["cbs"] / max(timing["n_cbs"], 1)
                print("timing ms/step: " + "  ".join(f"{k}={v:.1f}" for k, v in ms.items())
                    + f"  | cbs_block={cbs:.1f}ms each")


        tokens_seen += token_per_step

        # save checkpoint
        keep = bool(pending_at) and tokens_seen >= pending_at[0]
        roll = next_roll is not None and next_roll <= tokens_seen
        if keep or roll:
            path=f"{ckpt_config.save_dir}/tok_{tokens_seen}"
            save_checkpoint(path, model, opt, 
                            {"step": step, "tokens_seen": tokens_seen, "ema_g2": ema_g2, "ema_trS": ema_trS,
                             "wandb_id": wandb.run.id if dist_info.is_main else None},
                             progress, dist_info)
            if dist_info.is_main:
                if keep:
                    open(f"{path}/KEEP", "w").close()
                prune_checkpoints(ckpt_config.save_dir, ckpt_config.keep_last)
            dist.barrier() # every rank complete saving
            while pending_at and tokens_seen >= pending_at[0]:
                pending_at.pop(0)
            if next_roll is not None:
                next_roll = (tokens_seen // ckpt_config.ckpt_every + 1) * ckpt_config.ckpt_every if ckpt_config.ckpt_every else None
            last_time = time.perf_counter() # keep save time out of tokens/sec


        # log loss, token_seen, lr
        if measure:
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
                        "train/lr_muon": float(opt.param_groups[0]["lr"]),
                        "train/lr_adam": float(opt.param_groups[1]["lr"]),
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

        if log_config.profile:
            t_end = _t()