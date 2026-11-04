from lance.torch.data import SafeLanceDataset
from torch.utils.data import DataLoader, DistributedSampler
from flock.hub import s3
from transformers import AutoTokenizer

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

def load_dataset():
    s3.get_process(read_only=True)

    ds = SafeLanceDataset(
        "s3://flock-datasets/fineweb-edu",
        dataset_options={"version":1}
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sampler = DistributedSampler(ds, shuffle=True, drop_last=True)

    loader = DataLoader(
        dataset=ds,
        batch_size=32,
        sampler=sampler,
        drop_last=True,
        num_workers=4,
        multiprocessing_context="spawn",
        persistent_workers=True,
        pin_memory=True,
        collate_fn=Collator(tokenizer)
    )

    return loader