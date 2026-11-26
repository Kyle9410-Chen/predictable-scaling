import math
import random

import torch
import torch.distributed as dist

from flock.eval.model import ScoringModel
from flock.eval.tasks import SUITES


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

class PalomaEval:
    """flock's Paloma suite on the in-memory model, split across ranks."""

    def __init__(self, hf_model, tokenizer, dist_info, docs_per_source=None, window_batch=4):
        self.rank, self.world = dist_info.rank, dist_info.world_size
        self.device = dist_info.device
        self.sources = {s.name.split(":", 1)[1]: list(s.tasks) for s in SUITES["paloma"].members}
        tasks = [t for ts in self.sources.values() for t in ts]
        self.task_id = {t.name: i for i, t in enumerate(tasks)}
        self.window = tasks[0].window
        self.scorer = ScoringModel(hf_model, tokenizer, name="paloma",
                                   batch_tokens=self.window * window_batch)
        docs = []
        # The data arrangement is sources (many) > tasks (many) > documents (many) > 
        for tasks in self.sources.values():
            if docs_per_source is None:
                docs_per_task = None
            else:
                docs_per_task = max(1, math.ceil(docs_per_source / len(tasks)))
            
            for task in tasks:
                task_documents = task.load()
                if docs_per_task is not None and len(task_documents) > docs_per_task:
                    task_documents = random.Random(0).sample(task_documents, docs_per_task)
                docs += [(self.task_id[task.name], doc) for doc in task_documents]

        # make sure longest docs go first, so each shard get balanced load
        docs.sort(key=lambda task_and_doc: -len(task_and_doc[1].text))
        mine = docs[self.rank :: self.world]

        self.owner = torch.tensor([i for i, _ in mine])
        self.texts = [d.text for _, d in mine]
        self.ids = self.scorer.tokenize(self.texts)   # tokenized once
        self.nbytes = torch.tensor([len(t.encode()) for t in self.texts], dtype=torch.float64)

    def __call__(self):
        with torch.compiler.set_stance("force_eager"), torch.autocast("cuda", dtype=torch.bfloat16):
            s = self.scorer.rolling_loglikelihood(self.texts, self.window, self.ids)
        owner = self.owner.to(self.device)
        t = torch.zeros(len(self.task_id), 3, dtype=torch.float64, device=self.device)
        t[:, 0].index_add_(0, owner, s.logprob.double())
        t[:, 1].index_add_(0, owner, s.num_tokens.double())
        t[:, 2].index_add_(0, owner, self.nbytes.to(self.device))
        if dist.is_initialized() and self.world > 1:
            dist.all_reduce(t)
        bpb = (-t[:, 0] / (t[:, 2] * math.log(2))).tolist()
        out = {f"eval/paloma_{n}_bpb": sum(bpb[self.task_id[x.name]] for x in ts) / len(ts)
               for n, ts in self.sources.items()}
        out["eval/paloma_bpb"] = sum(out.values()) / len(out)
        out["eval/paloma_ppl"] = math.exp(min((-t[:, 0].sum() / t[:, 1].sum()).item(), 700))
        return out