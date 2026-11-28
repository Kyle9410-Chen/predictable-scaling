# In-Context Pretraining (ICLM) ordering for dolma3_mix_150B

Reorder the pretraining data so related documents share a context window, following
"In-Context Pretraining: Language Modeling Beyond Document Boundaries" (Shi et al., arXiv 2310.10638),
and compare against random grouping and plain one-document rows at the same token budget.

Code: `src/predictable_scaling/iclm.py` (one subcommand per step).
Prototype job: `src/predictable_scaling/sbatch/iclm_proto.sh`.

## Decisions

| Item | Choice |
|---|---|
| Cluster | 25a (this one): 8×H200 nodes, 112 CPUs, 1.9 TB RAM. Data is local; nothing goes through S3. The login node caps a user at 20 GB, so every heavy step is a Slurm job. |
| Source | `/work/benyang0509/datasets/dolma3_mix_150B_scored.lance`, version 2 (102,435,439 rows). Already decontaminated. |
| Eval | Paloma (as in `main.py`), plus in-context evaluations. No held-out split of the training data. |
| Embedding model | `facebook/contriever` (the paper's model), mean pooling, L2-normalised, first 512 tokens. |
| kNN | Exact inner-product search in torch on H200 (no faiss, no IVF-PQ). |
| Chain cap | 32,768 tokens (4 windows of 8,192), like the paper's concatenate-then-chop. |
| Repeats | Keep AI2's repeats (copies of an id), so the Dolma 3 mix is unchanged. |
| τ (dedup cutoff) | Chosen from the prototype and full-scale similarity statistics (step 3). |

## Facts the design rests on (verified)

- Rows pass the filter when `mtld` and `flesch_reading_ease` are both finite: 101,251,806 rows, 152.76B tokens.
- `n_tokens` = Llama-3.2-1B token count incl. BOS, +1 for the EOS the packer appends; identical for all copies of an id.
- Unique ids: 34,953,857 documents, 61.61B tokens (40%). Copies per id:
  1: 13.50M, 2: 6.48M, 3: 4.04M, 4: 3.52M, 5: 1.01M, 6: 2.50M, 7: 2.84M, 8: 0.86M, 9: 0.19M.
- Unique-only would change the mix (common_crawl 77% → 65%, olmocr_science_pdfs 12.5% → 27%, finemath 2.7% → 1.5%), which is why repeats are kept.
- A merged row `a + eos_token + bos_token + b` tokenizes to exactly `BOS a EOS BOS b`, the same ids as two
  separate rows (tested with trailing spaces, newlines and literal special-token strings). So a merged row's
  `n_tokens` is the sum of its members'.
- `PackedDataset` gives each row positions 0,1,2,… and resets them to 0 in each 8,192 window a row spills into
  (`dataloader.py:108-112`); `packed_collate` starts a flash-attention segment at every position 0. A merged
  row is therefore one attention segment per window, with no training-code change.
- `main.py` already has `--no-shuffle` (no fragment shuffle, no row shuffle buffer, no skipped fragments) and
  `--dataset-version`.

## Pipeline

Working directory: `/work/benyang0509/iclm/` (prototype in `proto/`).

### 0. Document table (CPU) — `iclm.py doctable`
Apply the row filter and keep the first occurrence of each id.
Output `doctable.parquet`, one row per unique document in source order: `src_pos, id, copies, n_tokens`.
Row number = `doc_idx`, used by every later step. Check: Σ `n_tokens × copies` = 152.76B.

### 1. Embed (GPU) — `iclm.py embed` under torchrun
Contriever, bf16, texts cut to 4,096 characters, then truncated to 512 tokens; batches are sorted by
length and bounded by tokens to reduce padding. Output fp16 chunks `emb/NNNNN.npy` of 65,536 documents
(35.0M × 768 × 2 B ≈ 54 GB). Chunks are split across GPUs and skipped if present, so a job can be resumed.
Estimate: a few H200-hours; reading and tokenizing may be the bottleneck.

### 2. Exact kNN (GPU) — `iclm.py knn` under torchrun
Every GPU holds the whole fp16 table (fits in 141 GB) and takes every world-th query block of 8,192.
Per database chunk of 2^20: `q @ db.T` (fp16), mask the query itself, `topk`, merge into the running top-k.
Neighbour scores are then recomputed in fp32 and re-sorted. Output `knn/I_NNNNN.npy` (int32) and
`knn/D_NNNNN.npy` (fp32), k = 10, one file pair per block (resumable).
Estimate: 1.9e18 FLOPs, about 1–3 H200-hours (10–25 min on 8 GPUs).

### 3. Near-duplicate removal (CPU) — `iclm.py simstats`, then `iclm.py dedup`
`simstats` reports the nearest-neighbour similarity histogram, the share of documents and tokens dedup
would drop at a range of τ, the similarity of random pairs, and writes sample pairs in similarity bands
to Markdown for reading. Pick τ from that.
`dedup`: every neighbour pair with similarity > τ is an edge; walk documents in `doc_idx` order and drop a
document linked to an earlier kept one. Output `keep.npy`.
Paper ablation: without dedup, Wikipedia PPL 8.3 vs 7.3 with it.

### 4. Chains (CPU) — `iclm.py chains`
Kept documents only. Degree = in-degree in the kNN graph among kept documents. Start at the unvisited
document with the lowest degree, move to the most similar unvisited neighbour, and at a dead end start a
new chain from the next lowest-degree unvisited document. When adding a document would pass 32,768 tokens,
start a new chain but continue the same walk. Output `chains.parquet`: `doc_idx, chain_id, chain_pos`.
Variants (`--per-pass`, `--max-adj`), added after the first training runs: `iclm_k10` was ~0.015–0.02 Paloma
bpb worse than `single` and `randgroup` (which matched each other) and its loss was noisier. Two causes
specific to how the chains were built:
- Repeated partners: deriving pass j by restricting the chain to copies >= j means an upsampled document
  meets the same neighbours, in the same order, in every pass (on the 1M prototype 51.5% of adjacent pairs
  repeat across rows). `--per-pass` runs a separate walk for each pass over the documents with copies >= j
  and never makes two documents adjacent again if they already were; output gets a `pass_` column and
  `randgroup`/`write` handle it.
- Near-duplicates side by side: dedup removes pairs above τ = 0.94, but 3.9% of adjacent pairs were in
  0.85–0.94 (templated code, SQL dumps, reworded SEO pages). `--max-adj 0.85` never steps to such a neighbour.
Build: `K=32 PERPASS=1 MAXADJ=0.85 DATASETS="iclm randgroup" sbatch iclm_build.sh`
→ `dolma3_mix_150B_{iclm,randgroup}_k32_pp_a0.85_t0.94.lance` (same documents and tokens; `single_t0.94`
remains the baseline).

`iclm.py inspect` reports chain statistics, adjacent-document similarity against a random grouping with the
same lengths, and writes sample chains to Markdown.

### 5. Repeats (CPU) — to build after the prototype
Pass j (j = 1..9) is every chain restricted, in the same order, to documents with `copies ≥ j`.
A training row is one (chain, pass). This keeps AI2's mix and never puts two copies in one row.

### 6. Write merged datasets (CPU) — to build after the prototype
Pass 1 reads the source in order and sends each kept document's text to the bucket (hash of row id, 128
buckets) of every row it belongs to. Pass 2 groups each bucket by row, sorts by `chain_pos`, joins with
`eos_token + bos_token`, shuffles, and writes 32,768-row fragments.
Same pattern as `_bucket_shuffle` / `_gather_bucket` in `preprocess.py`.

Columns (same schema for `iclm`, `randgroup` and `single`):

| Column | Type | Meaning |
|---|---|---|
| `text` | string | member texts joined with `eos_token + bos_token` |
| `n_tokens` | int32 | sum of members' `n_tokens` (exact: merged and separate tokenization give the same ids) |
| `n_docs` | int16 | number of documents in the row (1 for `single`) |
| `chain_id` | int64 | ICLM chain or random group the row came from |
| `pass_` | int8 | repeat pass j (1–9) |
| `ids` | list\<string\> | member ids, in order |
| `doc_n_tokens` | list\<int32\> | each member's length, so a row can be split back into documents |
| `subset` | list\<string\> | each member's subset, for checking the mix |

Dropped: the per-document scores (`mtld`, `flesch_reading_ease`, `fineweb_edu`, `fineweb_edu_int`) and the
other metadata (`source`, `added`, `created`, `version`, `metadata`, `extra`, `attributes`). A score is a
property of one document, not of a merged row; no row-level aggregate is stored, so score-based code
(curriculum sorting, `analysis.py`) cannot treat a merged row as one document. Any per-document field can be
recovered by joining `ids` back to `dolma3_mix_150B_scored`. Training reads only `text`.

### 7. Controls (CPU) — same writer, same kept documents, same repeats — `iclm.py randgroup`
- `randgroup`: the ICLM rows with their members swapped for random documents of about the same length.
  For each pass j, the pass-j documents are sorted by length (random tie-break), cut into blocks of 64,
  permuted inside each block, and every ICLM member is replaced by its image. Each control row has the
  same number of documents and nearly the same token length as its ICLM row; each document is still in
  exactly one row per pass. (Cutting a shuffled list to the ICLM token lengths was tried first and gave
  rows ~20% longer than ICLM, because ICLM has many short one-document rows; grouping once before the
  pass expansion was worse still, since related documents share copy counts.)
- `single`: one document per row, shuffled. Rebuilt rather than reusing `train.lance`, so that dedup
  (step 3) is identical across the three runs.

`iclm.py check` verifies each dataset: token total, `n_docs`/`n_tokens` consistent with the lists, no id
twice in a row, every kept document exactly `copies` times, and re-tokenized length + 1 = `n_tokens` on
sample rows.

Job: `src/predictable_scaling/sbatch/iclm_build.sh` (CPU; 1 GPU is requested only to get a node).
Outputs: `/work/benyang0509/datasets/dolma3_mix_150B_{iclm_k10_t0.94,randgroup_k10_t0.94,single_t0.94}.lance`.

## Experiment

Same model and config, 30B tokens, `--no-shuffle` and a pinned `--dataset-version` for all three:

1. `single`
2. `randgroup`
3. `iclm`

2 vs 3 isolates the ordering; 1 vs 2 isolates attention across document boundaries.
Eval: Paloma (existing), plus in-context evaluations with lm-eval (`/work/benyang0509/venvs/lmeval`):
few-shot classification (SST-2, AGNews, … at several shot counts), reading comprehension (RACE, BoolQ,
SQuAD), and the tasks already cached (ARC, HellaSwag, PIQA, LAMBADA, …).

## Checks before training

- In every dataset each kept id appears exactly `copies` times; total tokens = step 0 total minus the
  tokens dedup removed.
- Adjacent-document similarity: `iclm` much higher than `randgroup`; `randgroup` ≈ random pairs.
- Token-length distributions of `iclm` and `randgroup` rows match.
- Read about 20 ICLM chains by eye.

## Prototype (now)

1M unique documents sampled from the document table, steps 0–4 on one H200 node (`iclm_proto.sh`):
measure embedding docs/s and kNN throughput, read the τ report and sample chains. Extrapolate kNN time as
(prototype per-GPU time) × 35²/8. Neighbours are sparser at 1M than at 35M, so the final τ is set from the
full-scale `simstats`.

## About the official repo (github.com/swj0419/in-context-pretraining)

Reference only: it has no LICENSE (do not copy code), hard-codes the authors' paths, and has likely crashes
(`file_name` undefined in `sort_class.__init__`, `import faiss` commented out, an `input()` that blocks Slurm).
Its traversal differs from the paper (chains of at most 21 documents from kNN ranks 1–9, short chains merged
into a neighbour's), and its dedup compares only the previous document's first 100 characters.
Paper scale: 235M documents, kNN 6 h on 32 GPUs, traversal 12 h on 20 CPUs, 8,192-token windows.

## Risk

The paper's gains are at 7B, and reading-comprehension gains appeared after ~150B tokens. A 150M model at
30B tokens may show little beyond the in-context evaluations.
