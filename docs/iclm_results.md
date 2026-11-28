# In-Context Pretraining (ICLM) at 150M / 30B tokens — results

Does ordering pretraining data so that related documents share a context window (Shi et al., 2023,
"In-Context Pretraining", arXiv 2310.10638) help a small model? We built ICLM-ordered versions of
dolma3_mix_150B, trained 150M models on them and on two controls at the same 30B-token budget, and measured
(1) standalone language modelling (Paloma and three other corpora), (2) long-context completion (LAMBADA),
(3) how much each model gains from preceding related documents, and (4) downstream multiple-choice and
few-shot tasks with probability-based metrics.

**Summary.**
- **Long-context use improves.** Both ICLM models are better than the matched random-grouping controls and the
  one-document baseline on LAMBADA (+1.6 to +3.3 percentage points of accuracy), the only independent task that needs a
  whole passage to predict the next word. On held-out ICLM rows they gain 3.3× more from preceding related
  documents than the controls (+0.050 vs +0.015 nats/token) — a measure that matches ICLM's training
  objective, so it is supporting evidence only.
- **Standalone language modelling is not harmed by ICLM as such.** The first ICLM variant (`iclm_k10`) is
  slightly *better* than the baseline on wikitext, C4 and Pile bits per byte (−0.003 to −0.004, significant).
  The second, "fixed" variant (`iclm_k32_pp_a0.85`) is worse (+0.005 to +0.008), matching its Paloma result
  (0.008 bpb worse). The cost therefore comes from what changed between the two variants (§2.4), not from the
  ICLM ordering itself; which of the three changes is responsible is untested.
- **Multiple-choice benchmarks cannot tell the models apart** at 150M, even with probability metrics and paired
  tests; none of the models shows measurable few-shot learning.
- ICLM models are **hurt by unrelated preceding text** (they learn to assume context is related), whereas the
  random-grouping models ignore it.

Plan and pipeline details: [`iclm_plan.md`](iclm_plan.md). Code: `src/predictable_scaling/iclm.py`.
Evaluation scripts and raw results: `/work/benyang0509/iclm/bench/`.

---

## 1. What we compared

All datasets contain exactly the same text — 34.17M documents kept after near-duplicate removal, each repeated
as many times as AI2 upsampled it (`copies`, 1–9), **148.108B tokens** — and differ only in which documents
share a row. A row is one attention segment in training (positions restart at 0 for every row), so documents in
the same row can attend to each other.

| Dataset | A row is | Rows | Docs / row | Tokens / row (median / mean) |
|---|---|---|---|---|
| `single_t0.94` | one document | 98.8M | 1.00 | 693 / 1,499 |
| `randgroup_k10_t0.94` | random documents, same structure as `iclm_k10` | 32.6M | 3.03 | 1,404 / 4,547 |
| `iclm_k10_t0.94` | a chain of similar documents | 32.6M | 3.03 | 1,404 / 4,547 |
| `randgroup_k32_pp_a0.85_t0.94` | random documents, same structure as `iclm_k32pp` | 25.8M | 3.82 | 1,520 / 5,735 |
| `iclm_k32_pp_a0.85_t0.94` | chain, with the changes in §2.4 | 25.8M | 3.82 | 1,520 / 5,735 |

The `randgroup` control takes each ICLM row and swaps every member for a random document of about the same
length (median row-length difference: 0 tokens), so `randgroup` vs `iclm` isolates the ordering and
`single` vs `randgroup` isolates attention across document boundaries.

Training: 150M Llama-style model (6 layers, width 768), Muon, power-cool LR schedule, 8 × H200, batch
32 × 8,192 tokens, 30B tokens, same config and seed for every run; only `--dataset` and the run name differ.
One training run per dataset (no seed replicates).

## 2. How the ICLM data was built

### 2.1 Documents (step 0)
Source: `dolma3_mix_150B_scored.lance` v2 (102.4M rows). Rows with finite `mtld` and `flesch_reading_ease`
(101.25M rows, 152.76B tokens) were collapsed to one row per id: **34.95M unique documents, 61.6B unique
tokens**, with `copies` recording AI2's upsampling. Copies of an id have identical text and `n_tokens`.

### 2.2 Embeddings and exact nearest neighbours (steps 1–2)
- `facebook/contriever`, first 512 tokens, mean pooling, L2-normalised; 34.95M × 768 fp16 (54 GB);
  22 min on 8 × H200 (~3.5k docs/s per GPU).
- **Exact** inner-product top-k (no approximate index): every GPU holds the full table, query blocks of
  8,192 against database chunks of 2²⁰, then exact fp32 re-scoring. ~27 min on 8 × H200 for k = 10 or 32.

### 2.3 Near-duplicate removal and chains (steps 3–4)
- **Dedup, τ = 0.94:** a document is dropped if it is more similar than τ to an earlier kept document.
  Keeps 34.17M documents (97.8%), 148.1B of 152.8B tokens. 1% of documents have an exact (similarity 1.0)
  twin under another id.
- **Chain walk (paper's greedy traversal):** start at the unvisited document with the lowest in-degree, step to
  the most similar unvisited neighbour, start a new chain at a dead end or when the next document would exceed
  **32,768 tokens** (4 windows).
- **Repeats:** a document with `copies = c` appears in c rows ("passes" 1..c), never twice in one row.
- Rows are shuffled at write time; joined as `doc EOS BOS doc …`, which tokenizes to exactly the same ids as
  separate rows (verified), so a row's `n_tokens` is the sum of its members'.

| Chains | k = 10 | k = 32 |
|---|---|---|
| Docs per chain (mean / median / p90) | 4.09 / 2 / 9 | 6.46 / 3 / 18 |
| Tokens per chain (mean / median) | 7,164 / 2,027 | 11,300 / 3,438 |
| Adjacent-document similarity (mean) | 0.604 | 0.597 |
| Random grouping, same lengths | 0.191 | 0.191 |

### 2.4 The second ("fixed") variant
The first ICLM run (`iclm_k10`) had a noisier training loss and was ~0.015–0.02 bpb worse on Paloma before the
LR decay. Two properties of its construction looked suspicious:
1. **Repeated partners.** Later passes were derived by restricting a chain to members with `copies ≥ j`, so an
   upsampled document met the same neighbours, in the same order, in every pass (51.5% of adjacent pairs
   repeated across rows on a 1M-document prototype).
2. **Near-duplicates side by side.** Dedup only removes pairs above 0.94; 3.9% of adjacent pairs were in
   0.85–0.94 (templated code, SQL dumps, reworded SEO pages).

`iclm_k32_pp_a0.85` changes **three** things at once: a separate walk per pass that never makes two documents
adjacent twice (72.96M distinct adjacent pairs, none repeated), a 0.85 cap on adjacent similarity, and k = 32
instead of 10. Adjacent similarity 0.579; rows ~14% shorter than plain `iclm_k32`. In hindsight (§3) this
variant is the *worse* one on standalone text; the three changes have not been tested separately.

Every dataset passed the same checks: token total = 148,108,459,568; every kept document appears exactly
`copies` times; no id twice in a row; `n_docs` / `n_tokens` consistent with the per-document lists;
re-tokenized length + 1 = `n_tokens` on 300 sampled rows.

## 3. Results

Evaluation setup: lm-evaluation-harness 0.4.13 (`hf` backend), **float32** (a first pass in bfloat16 was
discarded: lm-eval sums token log-probs in the model dtype, which rounds long-document sums), BOS added,
max length 8,192, per-sample logging.

**Statistical method.** Two layers, kept separate:
- *Benchmark scores* (accuracy, perplexity, bits per byte, `acc`/`acc_norm`) and their **± values** are
  lm-eval's standard outputs. The ± is lm-eval's standard error of one model's score over the test items
  (for accuracy √(p(1−p)/(n−1))) — one standard error, not a 95% interval. Use these when comparing to
  published numbers.
- *Model-vs-model differences* are our addition, not part of any benchmark: a **paired bootstrap**. Each
  model is evaluated once; both models' per-item results on the same items are kept; the items are resampled
  with replacement 2,000 times and the difference A − B is recomputed on each resample (for bits per byte,
  as −Σ log-likelihood / (Σ bytes · ln 2) per model). The 2.5th–97.5th percentiles give the 95% interval in
  brackets; **`*` = the interval excludes 0**. Pairing removes item difficulty shared by both models, so a
  real difference can be significant even when the two models' ± bars overlap.
- Neither layer covers training-seed variance: there is one training run per dataset. With many comparisons,
  a few `*` are expected by chance; we rely on patterns that repeat across tasks or variants.

### 3.1 Paloma (training-time eval)
Read from the W&B dashboard (`eval/paloma_bpb`):

| Run | vs `single` |
|---|---|
| `randgroup_k10`, `randgroup_k32pp` | ≈ equal throughout |
| `iclm_k10` | ~0.015–0.02 bpb worse before LR decay; gap closes during decay |
| `iclm_k32pp` | gap shrinks during decay; **ends 0.008 bpb worse** |

ICLM's *training* loss is lower than the others', but that is not comparable: most ICLM tokens are predicted
with related documents in context.

### 3.2 Standalone language modelling on other corpora
Bits per byte, whole documents scored with rolling 8,192-token windows (wikitext: 62 documents; C4 and
pile_10k: 2,000 documents each). Lower is better.

| Model | wikitext | C4 | pile_10k |
|---|---|---|---|
| `single` | 0.9488 | 1.0802 | 1.0315 |
| `randgroup_k10` | 0.9472 | 1.0784 | 1.0290 |
| `iclm_k10` | **0.9460** | **1.0758** | **1.0273** |
| `randgroup_k32pp` | 0.9479 | 1.0798 | 1.0308 |
| `iclm_k32pp` | 0.9571 | 1.0849 | 1.0361 |

Paired difference A − B in bpb (negative = A better), 95% paired-bootstrap interval in brackets,
`*` = interval excludes 0 (see Statistical method):

| A − B | wikitext | C4 | pile_10k |
|---|---|---|---|
| `iclm_k10` − `single` | −0.0028* [−0.0048, −0.0009] | −0.0044* [−0.0067, −0.0028] | −0.0042* [−0.0050, −0.0034] |
| `iclm_k10` − `randgroup_k10` | −0.0012 [−0.0029, +0.0005] | −0.0026* [−0.0047, −0.0012] | −0.0017* [−0.0023, −0.0008] |
| `iclm_k32pp` − `single` | +0.0083* [+0.0069, +0.0098] | +0.0047* [+0.0029, +0.0059] | +0.0046* [+0.0030, +0.0069] |
| `iclm_k32pp` − `randgroup_k32pp` | +0.0093* [+0.0079, +0.0108] | +0.0051* [+0.0032, +0.0064] | +0.0053* [+0.0033, +0.0081] |
| `randgroup_k10` − `single` | −0.0017* [−0.0034, −0.0002] | −0.0018* [−0.0025, −0.0012] | −0.0025* [−0.0035, −0.0020] |
| `randgroup_k32pp` − `single` | −0.0010 [−0.0022, +0.0002] | −0.0004 [−0.0011, +0.0004] | −0.0007* [−0.0016, −0.0001] |

The two ICLM variants move in **opposite directions**, consistently across corpora and consistent with Paloma.
The cost seen for `iclm_k32pp` is therefore not a property of ICLM ordering in general. The difference between
the two ICLM runs (~0.009 bpb) is several times the spread among the three non-ICLM runs (≤ 0.0025), but with
one seed per dataset we cannot fully rule out run-to-run variance.

### 3.3 Long-context completion (LAMBADA) and grammar (BLiMP)
LAMBADA asks for the last word of a passage that can only be predicted from the whole passage (5,153 passages;
two versions of the text). **Accuracy** = share of passages where the model's greedy prediction of the whole
last word is exactly right; **perplexity** = exp of the mean negative log-likelihood of the last word. BLiMP:
share of 67,000 grammatical / ungrammatical sentence pairs where the grammatical one is more likely.

**Scores** (± standard error):

| Model | LAMBADA (OpenAI) acc ↑ | ppl ↓ | LAMBADA (standard) acc ↑ | ppl ↓ | BLiMP acc ↑ |
|---|---|---|---|---|---|
| `single` | 30.7% ± 0.6 | 39.8 | 23.7% ± 0.6 | 115.9 | 79.3% ± 0.1 |
| `randgroup_k10` | 32.5% ± 0.7 | 37.2 | 23.4% ± 0.6 | 111.8 | 79.2% ± 0.1 |
| `iclm_k10` | **34.1% ± 0.7** | **35.1** | 24.3% ± 0.6 | 115.8 | 79.2% ± 0.1 |
| `randgroup_k32pp` | 30.3% ± 0.6 | 42.4 | 22.3% ± 0.6 | 125.4 | 78.6% ± 0.1 |
| `iclm_k32pp` | 33.5% ± 0.7 | 36.3 | **24.7% ± 0.6** | **100.2** | **79.5% ± 0.1** |

**Paired differences in accuracy** A − B, with both models' scores, in percentage points (absolute; e.g.
30.7% → 34.1% is +3.4 points); 95% paired-bootstrap interval in brackets, `*` = interval excludes 0:

| A − B | LAMBADA (OpenAI): A, B → difference | LAMBADA (standard): A, B → difference |
|---|---|---|
| `iclm_k10` − `randgroup_k10` | 34.1%, 32.5% → **+1.6*** [+0.5, +2.7] | 24.3%, 23.4% → +0.8 [−0.1, +1.7] |
| `iclm_k32pp` − `randgroup_k32pp` | 33.5%, 30.3% → **+3.2*** [+2.0, +4.3] | 24.7%, 22.3% → **+2.4*** [+1.5, +3.3] |
| `iclm_k10` − `single` | 34.1%, 30.7% → **+3.3*** [+2.3, +4.4] | 24.3%, 23.7% → +0.6 [−0.2, +1.5] |
| `iclm_k32pp` − `single` | 33.5%, 30.7% → **+2.8*** [+1.7, +3.8] | 24.7%, 23.7% → **+1.1*** [+0.2, +2.0] |
| `randgroup_k10` − `single` | 32.5%, 30.7% → +1.8* [+0.7, +2.8] | 23.4%, 23.7% → −0.2 [−1.1, +0.7] |
| `randgroup_k32pp` − `single` | 30.3%, 30.7% → −0.4 [−1.5, +0.6] | 22.3%, 23.7% → −1.3* [−2.2, −0.4] |

Both ICLM models beat their matched controls on LAMBADA, including the variant that is worse on standalone
text — the clearest independent sign that ICLM improves use of long context. (The paired differences are
computed per passage, so they can differ by ±0.1 point from the difference of the rounded scores above.) BLiMP
differs by less than 1 point across all models; at ±0.1 standard error the 0.9-point gap between `iclm_k32pp` and
`randgroup_k32pp` is nominally beyond noise but tiny, and grammar is not where ICLM is expected to act.

### 3.4 Context gain on held-out ICLM rows (matches the training objective — supporting evidence only)

2,000 held-out rows of the fixed ICLM dataset from the 2 fragments the training dataloader never reads (shuffle
seed 0, `num_eval_fragment = 2`); rows with ≥ 2 documents and ≤ 8,192 tokens. Target = each row's last
document; loss on its tokens (nats/token, incl. EOS), computed in float32:
- **alone** — `BOS target EOS`;
- **related** — the row's own preceding documents, merged exactly as in training (`doc EOS BOS …`), then the
  target (mean 1,694 context tokens);
- **random** — the preceding documents of another held-out row with similar context length, then the target.

| Model | Alone | Related (gain) | Random (gain) | Docs helped by related / random |
|---|---|---|---|---|
| `single` | **2.877** | 5.189 (−2.31) | 5.369 (−2.49) | 0% / 0% |
| `randgroup_k10` | 2.879 | 2.865 (+0.015) | 2.882 (−0.003) | 76% / 34% |
| `randgroup_k32pp` | 2.882 | 2.867 (+0.015) | 2.884 (−0.002) | 80% / 40% |
| `iclm_k10` | 2.899 | **2.849 (+0.050)** | 2.927 (−0.028) | 91% / 3% |
| `iclm_k32pp` | 2.900 | **2.850 (+0.050)** | 2.928 (−0.029) | 93% / 3% |

Caveats: the "related" context is ICLM's own construction (same Contriever neighbours), so this is
in-distribution for ICLM models by design; the meaningful comparison is ICLM vs `randgroup`, both trained on
merged rows of the same structure. `single` never attended across a document boundary and breaks down on any
multi-document input. The targets are not Paloma-style text, which is why "alone" ranks the models differently
from §3.2. ICLM models are hurt by unrelated context (−0.03); the controls ignore it.

### 3.5 Multiple-choice and few-shot tasks

**Tasks.** 0-shot: ARC-easy, ARC-challenge, HellaSwag, PIQA, SciQ, WinoGrande, OpenBookQA, COPA, BoolQ, RACE,
RTE, SST-2. Few-shot: SST-2 (4, 16), RTE (4, 16), BoolQ (4), ARC-easy / PIQA / HellaSwag (5). ≤ 2,000 items
per task, same few-shot sampling seed for every model.

**Metrics** (lm-eval scores an option by the log-likelihood ℓᵢ of its text given the prompt; `acc` takes the
argmax of ℓᵢ, `acc_norm` the argmax of ℓᵢ / characters). Because 0/1 accuracy is a weak signal for small
models (cf. DataDecide), we also report, with qᵢ = exp(ℓᵢ / charsᵢ) and c the correct option: correct prob q_c,
total prob Σqᵢ, **normalized correct prob** q_c / Σqᵢ, **margin** q_c − max_{i≠c} qᵢ, and log-prob per
character of the correct answer.

**0-shot macro means over 12 tasks:**

| Model | acc_norm | Normalized correct prob | Margin |
|---|---|---|---|
| `single` | .4839 | .3923 | −.0035 |
| `randgroup_k10` | .4911 | .3913 | −.0058 |
| `iclm_k10` | .4787 | .3932 | −.0052 |
| `randgroup_k32pp` | .4862 | .3918 | −.0051 |
| `iclm_k32pp` | .4785 | .3916 | −.0043 |

**Paired normalized correct prob, 20 task/shot settings per pair:**

| Pair | Mean difference | Settings significantly better | Significantly worse |
|---|---|---|---|
| `iclm_k10` − `randgroup_k10` | +0.0020 | 7 | 4 |
| `iclm_k32pp` − `randgroup_k32pp` | −0.0006 | 2 | 4 |
| `iclm_k32pp` − `single` | −0.0006 | 2 | 2 |
| `randgroup_k32pp` − `single` | −0.0000 | 4 | 3 |

With 324 paired tests, ~16 "significant" results are expected by chance, the control pair shows the same
scatter, and the significant tasks mostly do not replicate across the two ICLM variants. The models are
indistinguishable on these tasks. Few-shot learning (0 → k-shot change in normalized correct prob on the same
items) is within ±0.003 for every model and task, and BoolQ gets worse with examples for all of them: these
150M models do essentially no few-shot learning, so ICLM has nothing to improve there.

Not run: `ag_news`, `dbpedia_14` (need `unitxt` in the lm-eval venv), `coqa` (its scorer imports a module
removed in transformers 5.x).

## 4. Engineering notes

- **Throughput.** Merged-row datasets trained at ~1.2M tok/s vs ~2M for `single`. Not GPU compute (on one GPU
  the step is only 4% slower with merged rows) and not the node: `PackedDataset` tokenizes 256 *rows* at a
  time, which for merged rows is ~1.2M tokens (~2 s) per call — longer than the ~1 s of prefetched batches
  covers (4 workers × `prefetch_factor` 2), so ranks stall and DDP makes every GPU wait. Measured: 16% of time
  waiting per process at current settings; 0.7% with `tokenize_batch=32`, 1.2% with `prefetch_factor=8`. The
  packed windows are identical with either change.
- **lm-eval precision.** Use `dtype=float32` for likelihood-based comparisons; in bfloat16 the per-document
  log-likelihood sums are rounded (logged values like −1360.0).
- **Run naming.** `iclm_k10` (job 868646) trained under the old name `warmup-flesch-600M-dolma3mix-150M-1.5B`
  (W&B run name and checkpoint); its checkpoint overwrote the earlier flesch checkpoint of that name and has been
  renamed to `models/iclm-k10-t0.94-dolma3mix-150M-30B`.
- **Reproducing §3.2, §3.3, §3.5:** `sbatch src/predictable_scaling/sbatch/benchmark.sh`
  (`src/predictable_scaling/benchmark.py`: `evaluate`, `language-modeling`, `multiple-choice`).
- **Cost of the pipeline.** On 8 × H200: embed 22 min, exact kNN 27 min, dedup + chains + report ~20 min, each
  dataset write + check ~15–18 min (CPU).

## 5. Conclusions and next steps

1. At 150M / 30B tokens, ICLM ordering improves the use of long context (LAMBADA accuracy +1.6 to +3.3 percentage points over
   matched controls) and, in its original form (`iclm_k10`), slightly *improves* standalone language
   modelling (−0.003 to −0.004 bpb). It does not change multiple-choice or few-shot results, which are
   uninformative at this size.
2. The standalone-text cost seen on Paloma comes from the second variant, which changed three things at once.
   Ablate them — per-pass walk, 0.85 adjacency cap, k = 32 — one at a time against `iclm_k10`, and add a second
   seed for `single` and `iclm_k10` to measure run-to-run variance.
3. Prefer `iclm_k10`-style data (single walk, repeats derived from the chain) unless an ablation shows a
   specific change helps.
4. ICLM models are hurt by unrelated preceding text; evaluate on mixed-relevance prompts (unrelated few-shot
   examples, retrieval with distractors) before using them where context quality is not controlled.
5. Whether the context gain turns into downstream in-context ability needs a larger model (≥ 1B).
6. Apply `tokenize_batch=32` in `load_dataset` before further merged-row training (~1.7× throughput).
