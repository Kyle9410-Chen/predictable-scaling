import lance
import random
import torch
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, get_worker_info
from flock.hub import s3
from transformers import AutoTokenizer
from itertools import islice, batched

class SequentialLanceDataset(IterableDataset):
    def __init__(self, uri, rank, world_size, version=1, column="text", 
                 buffer_size=4096, seed=0, read_batch_size=256, split="train", 
                 num_eval_fragment=2, eval_rows=2048, skip: dict[int, int] | None = None):
        if split not in ("train", "eval"):
            raise ValueError("split must be train or eval")
        
        self.uri: str = uri
        self.rank: int = rank
        self.world_size: int = world_size
        self.version: int = version
        self.column: str = column
        self.buffer_size: int = buffer_size
        self.seed: int = seed
        self.read_batch_size: int = read_batch_size
        self.split: str = split
        self.num_eval_fragment: int = num_eval_fragment
        self.eval_rows: int = eval_rows
        self.skip = skip # a dictionary of [shard: how many fragment to skip]

    def _read(self, fragments):
        for fragment in fragments:
            for batch in fragment.to_batches(columns = [self.column], batch_size = self.read_batch_size):
                yield from batch.to_pylist()
    
    def _rows(self, shard, num_shards):
        ds = lance.dataset(self.uri, version=self.version)
        fragments = ds.get_fragments()
        random.Random(self.seed).shuffle(fragments)

        if self.split == "train":
            frags = fragments[self.num_eval_fragment:][shard::num_shards]
            start = (self.skip or {}).get(shard, 0)
            for i in range(start, len(frags)):
                for row in self._read([frags[i]]):
                    row["_frag"] = i # make sure the index keep on after skip
                    yield row
        elif self.split == "eval":
            # num_eval_fragment may be smaller than number of workers. so we slice by rows to ensure each worker gets some.
            # and limit the number of total rows for evaluation. make sure total row same when scaling.
            rows = islice(self._read(fragments[:self.num_eval_fragment]), self.eval_rows)
            yield from islice(rows, shard, None, num_shards)
        else:
            raise ValueError("Invalid split: expected 'train' or 'eval' but got " + self.split)

    def __iter__(self):
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info else (0, 1)
        shard = self.rank * num_workers + worker_id
        rng = random.Random(self.seed + shard + 1)
        buffer, cur = [], 0
        for row in self._rows(shard, self.world_size * num_workers):
            cur = row.get("_frag", 0)
            if len(buffer) < self.buffer_size:
                buffer.append(row)
                continue
            j = rng.randrange(self.buffer_size)
            out = buffer[j]
            out["_shard"], out["_pos"] = shard, cur
            yield out
            buffer[j] = row
        rng.shuffle(buffer)
        for out in buffer:
            out["_shard"], out["_pos"] = shard, cur
            yield out

class PackedDataset(IterableDataset):
    def __init__(self, base: SequentialLanceDataset, tokenizer, seq_len=8192, column="text", tokenize_batch=256):
        self.base = base
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.column = column
        self.tokenize_batch = tokenize_batch

    def __iter__(self):
        eos = self.tokenizer.eos_token_id
        L = self.seq_len
        buffer: list[int] = []
        pos: list[int] = []
        for row in batched(self.base, self.tokenize_batch):
            shard, frag = row[-1]["_shard"], row[-1]["_pos"]
            enc = self.tokenizer([r[self.column] for r in row])["input_ids"]
            for ids in enc:
                ids.append(eos)
                buffer.extend(ids)
                pos.extend(range(len(ids)))
            n_full = len(buffer) // L
            for i in range(n_full):
                s = slice(i * L, (i+1) * L)
                p = torch.tensor(pos[s], dtype=torch.long)
                start = pos[i*L]
                if start != 0:
                    resets = (p[1:] == 0).nonzero()
                    end = resets[0].item() + 1 if len(resets) else L
                    p[:end] -= start
                yield {"input_ids": torch.tensor(buffer[s], dtype=torch.long), "position_ids": p, "shard": shard, "frag": frag}
            buffer, pos = buffer[n_full*L:], pos[n_full*L:]

class Collator:
    def __init__(self, tokenizer, column="text"):
        self.tokenizer = tokenizer
        self.column = column
    
    def __call__(self, rows):
        enc = self.tokenizer(
            [r[self.column] for r in rows],
            return_tensors = "pt",
            padding=True,
            truncation=True,
            max_length=8192
        )

        ids = enc["input_ids"]
        mask = enc["attention_mask"].bool()
        labels = ids.clone()
        labels.masked_fill_(~mask, -100)

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": labels
            }

def packed_collate(items):
    ids = torch.cat([x["input_ids"] for x in items]).unsqueeze(0)
    pos = torch.cat([x["position_ids"] for x in items]).unsqueeze(0)

    starts = (pos[0] == 0).nonzero().flatten()
    cu = torch.cat([starts, torch.tensor([pos.shape[1]])]).to(torch.int32)
    max_len = int(cu.diff().max())
    max_len = 1 << (max_len-1).bit_length() # round up to power of 2, make the kernel compile friendly

    return {
        "input_ids": ids,
        "labels": ids,
        "position_ids": pos,
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": max_len,
        "max_length_k": max_len,
        "data_pos": (items[-1]["shard"], items[-1]["frag"]) # supply the information to training loop for saving data progress when creating checkpoints
    }

def load_dataset(rank: int, world_size: int, split: str="train", workers: int=8, batch_size=4, seq_len=8192, skip: dict[int, int] = None) -> DataLoader:
    s3.get_process(read_only=True)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    loader: DataLoader = None

    # list evaluation set in memory prevent loading it every time
    if split == "eval":
        ds = SequentialLanceDataset(
                "s3://flock-datasets/fineweb-edu",
                rank=0,
                world_size=1,
                split="eval"
            )
        rows = list(ds)
        sampler = DistributedSampler(rows, num_replicas=world_size, rank=rank, shuffle=False)

        return DataLoader(
            dataset=rows,
            batch_size=batch_size,
            num_workers=workers,
            persistent_workers=True,
            sampler=sampler,
            pin_memory=True,
            collate_fn=Collator(tokenizer)
        )

    else:
        base = SequentialLanceDataset(
                "s3://flock-datasets/fineweb-edu",
                rank=rank,
                world_size=world_size,
                split="train",
                skip=skip
            )

        ds = PackedDataset(base, tokenizer, seq_len)

        loader = DataLoader(
            dataset=ds,
            batch_size=batch_size,
            num_workers=workers,
            multiprocessing_context="spawn",
            persistent_workers=True,
            pin_memory=True,
            collate_fn=packed_collate
        )

    return loader