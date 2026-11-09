# This checks chunk lengths, that no tokens are lost or reordered, where the positions reset, and the collator's cu_seq_lens.

import random, torch
from predictable_scaling.dataloader import PackedDataset, packed_collate

EOS, BOS = 0, 1

class FakeTok:
    eos_token_id = EOS
    def __call__(self, texts):
        return {"input_ids": [[BOS] + [int(t) for t in s.split()] for s in texts]}

def make_docs(n=200, seed=0):
    rng = random.Random(seed)
    return [{"text": " ".join(str(rng.randint(2, 999)) for _ in range(rng.randint(1, 40)))}
            for _ in range(n)]

def test_packed_dataset():
    L = 16
    docs = make_docs()
    chunks = list(PackedDataset(docs, FakeTok(), seq_len=L, tokenize_batch=7))

    # 1. every chunk exactly L
    assert all(c["input_ids"].numel() == L for c in chunks)

    # 2. no tokens lost / reordered: chunks are a prefix of the full stream
    stream = []
    for d in docs:
        stream += [BOS] + [int(t) for t in d["text"].split()] + [EOS]
    flat = torch.cat([c["input_ids"] for c in chunks]).tolist()
    assert flat == stream[:len(flat)]
    assert len(stream) - len(flat) < L          # only the tail remainder dropped

    # 3. positions reset exactly at doc starts and chunk starts
    for c in chunks:
        ids, pos = c["input_ids"].tolist(), c["position_ids"].tolist()
        for i in range(L):
            starts_doc = i == 0 or ids[i - 1] == EOS
            assert (pos[i] == 0) == starts_doc, (i, ids, pos)
            if not starts_doc:
                assert pos[i] == pos[i - 1] + 1

def test_collate():
    L, B = 16, 4
    chunks = list(PackedDataset(make_docs(), FakeTok(), seq_len=L))[:B]
    out = packed_collate(chunks)

    assert out["input_ids"].shape == (1, B * L)
    cu = out["cu_seq_lens_q"]
    assert cu.dtype == torch.int32
    assert cu[0] == 0 and cu[-1] == B * L
    assert (cu.diff() > 0).all()
    expected = (out["position_ids"][0] == 0).nonzero().flatten().tolist() + [B * L]
    assert cu.tolist() == expected
    assert out["max_length_q"] == int(cu.diff().max())
    assert isinstance(out["max_length_q"], int)

if __name__ == "__main__":
    test_packed_dataset(); test_collate(); print("data tests OK")
