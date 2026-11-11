import torch
import torch.distributed as dist

@torch.no_grad()
def eval(model, eval_loader: torch.utils.data.DataLoader, dist_info):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=dist_info.device)

    # the eval sequence change every time, so force eager mode
    with torch.compiler.set_stance("force_eager"):
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