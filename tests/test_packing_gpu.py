"""
Pack documents A and B into one row. 
B's hidden states should match running B alone,
    and should differ when you run the same row without 
    the masking arguments. 
The second comparison matters: without it, 
    a passing result could just mean the outputs happen to be close.
"""
import torch
from predictable_scaling.model import build_model
from predictable_scaling.dataloader import packed_collate

torch.manual_seed(0)
model = build_model().cuda().eval()
body = model.model                      # LlamaModel: skip lm_head / Liger loss

A = torch.randint(10, 100_000, (300,))
B = torch.randint(10, 100_000, (500,))
item = {"input_ids": torch.cat([A, B]),
        "position_ids": torch.cat([torch.arange(300), torch.arange(500)])}
batch = packed_collate([item])
batch.pop("labels")
batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
print("cu_seq_lens:", batch["cu_seq_lens_q"].tolist())   # expect [0, 300, 800]

@torch.no_grad()
def hidden(**kw):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return body(**kw).last_hidden_state[0].float()

ref_B   = hidden(input_ids=B[None].cuda())                   # B alone
packed  = hidden(**batch)[300:]                              # B inside packed row
no_mask = hidden(input_ids=batch["input_ids"])[300:]          # plain causal, no positions

d_packed  = (packed  - ref_B).abs().max().item()
d_nomask  = (no_mask - ref_B).abs().max().item()
print(f"packed vs alone : {d_packed:.4e}   (should be ~bf16 noise)")
print(f"no-mask vs alone: {d_nomask:.4e}   (should be much larger)")
assert d_packed < 0.1 * d_nomask, "doc masking NOT applied"

# Liger + labels path runs end to end; random-init loss should be ~ln(128256)=11.76
batch["labels"] = batch["input_ids"]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    print("loss:", model(**batch).loss.item())
