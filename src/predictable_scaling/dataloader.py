import os
import lance
import random

from lance.torch.data import LanceDataset
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, get_worker_info
from flock.hub import s3
from transformers import AutoTokenizer
from itertools import islice

class SequentialLanceDataset(IterableDataset):
    def __init__(self, uri, rank, world_size, version=1, column="text", 
                 buffer_size=4096, seed=0, read_batch_size=256, split="train", num_eval_fragment=2, eval_rows=2048):
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

    def _read(self, fragments):
        for fragment in fragments:
            for batch in fragment.to_batches(columns = [self.column], batch_size = self.read_batch_size):
                yield from batch.to_pylist()
    
    def _rows(self, shard, num_shards):
        ds = lance.dataset(self.uri, version=self.version)
        fragments = ds.get_fragments()
        random.Random(self.seed).shuffle(fragments)

        if self.split == "train":
            yield from self._read(fragments[self.num_eval_fragment:][shard::num_shards])
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
        buffer = []
        for row in self._rows(shard, self.world_size * num_workers):
            if len(buffer) < self.buffer_size:
                buffer.append(row)
                continue
            j = rng.randrange(self.buffer_size)
            yield buffer[j]
            buffer[j] = row
        rng.shuffle(buffer)
        yield from buffer

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
            max_length=4096
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
    
def load_dataset(rank: int, world_size: int, split: str="train"):
    s3.get_process(read_only=True)

    ds = SequentialLanceDataset(
        "s3://flock-datasets/fineweb-edu",
        rank=rank,
        world_size=world_size,
        split=split
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    loader = DataLoader(
        dataset=ds,
        batch_size=4,
        num_workers=8,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        collate_fn=Collator(tokenizer)
    )

    return loader