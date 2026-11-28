"""In-Context Pretraining (Shi et al. 2023, arXiv 2310.10638): order documents so related ones share a context.

One subcommand per step (see docs/iclm_plan.md):
    doctable  one row per unique id (first occurrence) of the filtered source
    sample    random subset of a document table (prototype)
    embed     Contriever embeddings, fp16 chunks           (GPU, launch with torchrun)
    knn       exact top-k inner-product neighbours          (GPU, launch with torchrun)
    simstats  similarity statistics and sample pairs, for choosing tau
    dedup     drop near-duplicates above tau
    chains    greedy lowest-degree walk into token-capped chains
    inspect   chain statistics and sample chains

A document is addressed by its row in the document table (doc_idx); every later file is in that order.
"""
import os
import shutil
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ---------------------------------------------------------------- helpers

def _dist():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))

def _save(path: Path, arr: np.ndarray):
    """Write atomically, so a file that exists is complete (steps resume by skipping existing files)."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)

def fetch_texts(ds: lance.LanceDataset, positions) -> list[str]:
    positions = np.asarray(positions, dtype=np.int64)
    u, inv = np.unique(positions, return_inverse=True)
    col = ds.take(u, columns=["text"])["text"].to_pylist()
    return [col[i] for i in inv]

def load_emb(emb_dir) -> np.ndarray:
    files = sorted(Path(emb_dir).glob("[0-9]*.npy"))
    return np.concatenate([np.load(f) for f in files])

def load_knn(knn_dir) -> tuple[np.ndarray, np.ndarray]:
    d = Path(knn_dir)
    I = np.concatenate([np.load(f) for f in sorted(d.glob("I_*.npy"))])
    D = np.concatenate([np.load(f) for f in sorted(d.glob("D_*.npy"))])
    return I, D

def _doctable(path, columns=None) -> pd.DataFrame:
    return pq.read_table(path, columns=columns).to_pandas()

# ---------------------------------------------------------------- step 0: document table

def build_doctable(src, version, out):
    """Filter rows (finite mtld and flesch_reading_ease), keep the first occurrence of each id."""
    ds = lance.dataset(src, version=version)
    H, P, T = [], [], []
    off, t0 = 0, time.time()
    scanner = ds.scanner(columns=["id", "n_tokens", "mtld", "flesch_reading_ease"], batch_size=1 << 20, scan_in_order=True)
    for b in scanner.to_batches():
        ok = pc.and_(pc.is_finite(b["mtld"]), pc.is_finite(b["flesch_reading_ease"]))
        idx = np.flatnonzero(ok.to_numpy(zero_copy_only=False))
        ids = b["id"].to_numpy(zero_copy_only=False)[idx]
        H.append(pd.util.hash_array(ids).view(np.int64))
        P.append(idx.astype(np.int64) + off)
        T.append(b["n_tokens"].to_numpy()[idx])
        off += b.num_rows
    h, pos, ntok = np.concatenate(H), np.concatenate(P), np.concatenate(T)
    print(f"scanned {off:,} rows, {len(h):,} pass the filter, {ntok.sum() / 1e9:.2f}B tokens ({time.time() - t0:.0f}s)", flush=True)

    _, first, inv, copies = np.unique(h, return_index=True, return_inverse=True, return_counts=True)
    if (ntok[first][inv] != ntok).any():
        raise ValueError("copies of an id differ in n_tokens")
    order = np.argsort(first)  # source order
    first, copies = first[order], copies[order]
    upos = pos[first]
    ids = ds.take(upos, columns=["id"])["id"]  # string ids of the kept rows only
    table = pa.table({
        "src_pos": upos,
        "id": ids,
        "copies": copies.astype(np.int8),
        "n_tokens": ntok[first].astype(np.int32),
    })
    if pc.count_distinct(table["id"]).as_py() != table.num_rows:
        raise ValueError("id hash collision")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)
    total = (ntok[first].astype(np.int64) * copies).sum()
    print(f"{table.num_rows:,} unique documents, {ntok[first].sum() / 1e9:.2f}B unique tokens, "
          f"{total / 1e9:.2f}B with copies -> {out}", flush=True)

def sample(doctable, n, out, seed=0):
    t = pq.read_table(doctable)
    idx = np.sort(np.random.default_rng(seed).choice(t.num_rows, size=n, replace=False))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t.take(idx), out)
    print(f"sampled {n:,} of {t.num_rows:,} documents -> {out}")

# ---------------------------------------------------------------- step 1: embed

def embed(doctable, src, version, out_dir, model_name="facebook/contriever", chunk=65_536,
          batch_tokens=262_144, max_len=512, max_chars=4096):
    import torch
    from transformers import AutoModel, AutoTokenizer

    rank, world, local = _dist()
    dev = torch.device("cuda", local)
    torch.cuda.set_device(dev)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = _doctable(doctable, ["src_pos"])["src_pos"].to_numpy()
    n_chunks = -(-len(pos) // chunk)
    todo = [c for c in range(rank, n_chunks, world) if not (out_dir / f"{c:05d}.npy").exists()]
    print(f"[rank {rank}] {len(todo)} of {n_chunks} chunks to embed", flush=True)
    if not todo:
        return

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="sdpa", dtype=torch.bfloat16).to(dev).eval()
    ds = lance.dataset(src, version=version)

    def prepare(c):
        """Read and tokenize one chunk; return length-sorted, token-bounded padded batches."""
        texts = [t[:max_chars] for t in fetch_texts(ds, pos[c * chunk:(c + 1) * chunk])]
        ids = tok(texts, truncation=True, max_length=max_len, padding=False,
                  return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        order = np.argsort(lens, kind="stable")
        batches, start = [], 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end + 1 - start) * lens[order[end]] <= batch_tokens:
                end += 1
            sel = order[start:end]
            L = int(lens[sel[-1]])
            x = np.full((len(sel), L), tok.pad_token_id, dtype=np.int64)
            for r, i in enumerate(sel):
                x[r, :lens[i]] = ids[i]
            m = np.arange(L)[None, :] < lens[sel][:, None]
            batches.append((sel, torch.from_numpy(x).pin_memory(), torch.from_numpy(m).pin_memory()))
            start = end
        return len(texts), int(lens.sum()), batches

    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(prepare, todo[0])
    t_all, docs_all = time.time(), 0
    for k, c in enumerate(todo):
        t0 = time.time()
        n, n_tok, batches = nxt.result()
        t_read = time.time() - t0
        if k + 1 < len(todo):
            nxt = pool.submit(prepare, todo[k + 1])
        out = torch.empty((n, model.config.hidden_size), dtype=torch.float16, device=dev)
        padded = 0
        with torch.inference_mode():
            for sel, x, m in batches:
                x, m = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True)
                h = model(input_ids=x, attention_mask=m).last_hidden_state.float()
                mf = m.unsqueeze(-1).float()
                e = (h * mf).sum(1) / mf.sum(1)
                out[torch.from_numpy(sel).to(dev)] = torch.nn.functional.normalize(e, dim=-1).half()
                padded += x.numel()
        _save(out_dir / f"{c:05d}.npy", out.cpu().numpy())
        dt = time.time() - t0
        docs_all += n
        print(f"[rank {rank}] chunk {c}: {n / dt:,.0f} docs/s, {n_tok / dt:,.0f} tok/s, "
              f"waited {t_read:.1f}s for data, padding {1 - n_tok / padded:.1%}", flush=True)
    print(f"[rank {rank}] done: {docs_all:,} docs in {time.time() - t_all:.0f}s "
          f"({docs_all / (time.time() - t_all):,.0f} docs/s)", flush=True)

# ---------------------------------------------------------------- step 2: exact kNN

def knn(emb_dir, out_dir, k=10, qblock=8192, dchunk=1 << 20):
    import torch

    rank, world, local = _dist()
    dev = torch.device("cuda", local)
    torch.cuda.set_device(dev)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(emb_dir).glob("[0-9]*.npy"))
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    N, dim = sum(s[0] for s in shapes), shapes[0][1]
    n_blocks = -(-N // qblock)
    todo = [b for b in range(rank, n_blocks, world) if not (out_dir / f"D_{b:05d}.npy").exists()]
    print(f"[rank {rank}] N={N:,} dim={dim}; {len(todo)} of {n_blocks} query blocks to search", flush=True)
    if not todo:
        return

    t0 = time.time()
    E = torch.empty((N, dim), dtype=torch.float16, device=dev)
    o = 0
    for f in files:
        a = torch.from_numpy(np.load(f))
        E[o:o + len(a)] = a.to(dev)
        o += len(a)
    print(f"[rank {rank}] loaded embeddings in {time.time() - t0:.0f}s", flush=True)

    t_all = time.time()
    with torch.inference_mode():
        for b in todo:
            qs, qe = b * qblock, min((b + 1) * qblock, N)
            q = E[qs:qe]
            best_v = best_i = None
            for off in range(0, N, dchunk):
                s = q @ E[off:off + dchunk].T
                lo, hi = max(qs, off), min(qe, off + s.shape[1])
                if lo < hi:  # the query itself
                    r = torch.arange(lo, hi, device=dev)
                    s[r - qs, r - off] = float("-inf")
                v, i = s.topk(k, dim=1)
                i += off
                if best_v is not None:
                    v, j = torch.cat([best_v, v], 1).topk(k, dim=1)
                    i = torch.cat([best_i, i], 1).gather(1, j)
                best_v, best_i = v, i
                del s
            # exact fp32 scores, re-sorted
            d = torch.einsum("qd,qkd->qk", q.float(), E[best_i].float())
            d, j = d.sort(dim=1, descending=True)
            best_i = best_i.gather(1, j)
            _save(out_dir / f"I_{b:05d}.npy", best_i.int().cpu().numpy())
            _save(out_dir / f"D_{b:05d}.npy", d.cpu().numpy())
    dt = time.time() - t_all
    nq = sum(min((b + 1) * qblock, N) - b * qblock for b in todo)
    rate = nq * N / dt
    print(f"[rank {rank}] {nq:,} queries in {dt:.0f}s: {rate:.3e} pairs/s per GPU; "
          f"35.0M x 35.0M on 8 GPUs would take {(35.0e6 ** 2) / rate / 8 / 3600:.2f} h", flush=True)

# ---------------------------------------------------------------- step 3: dedup

def dedup_mask(I, D, tau) -> np.ndarray:
    """Pairs above tau are edges; in doc_idx order, drop a document linked to an earlier kept one."""
    r, c = np.nonzero(D > tau)
    a, b = r, I[r, c].astype(np.int64)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    order = np.lexsort((lo, hi))  # every edge into a document comes after that document's own fate is set
    keep = np.ones(len(I), dtype=bool)
    for l, h in zip(lo[order].tolist(), hi[order].tolist()):
        if keep[l]:
            keep[h] = False
    return keep

def simstats(doctable, emb_dir, knn_dir, src, version, out, pairs_per_band=15, seed=0):
    rng = np.random.default_rng(seed)
    dt = _doctable(doctable, ["src_pos", "n_tokens", "copies"])
    I, D = load_knn(knn_dir)
    E = load_emb(emb_dir)
    N = len(I)
    tok = dt.n_tokens.to_numpy().astype(np.int64) * dt.copies.to_numpy()
    lines = [f"# Similarity statistics ({N:,} documents)\n"]

    a, b = rng.integers(0, N, 200_000), rng.integers(0, N, 200_000)
    rnd = (E[a].astype(np.float32) * E[b]).sum(1)
    q = [0.01, 0.1, 0.5, 0.9, 0.99]
    lines.append("Random pairs, quantiles " + ", ".join(f"p{int(x * 100)}={v:.3f}" for x, v in zip(q, np.quantile(rnd, q))))
    lines.append("Nearest neighbour, quantiles " + ", ".join(f"p{int(x * 100)}={v:.3f}" for x, v in zip(q, np.quantile(D[:, 0], q))))
    lines.append(f"Rank-{I.shape[1]} neighbour, quantiles " + ", ".join(f"p{int(x * 100)}={v:.3f}" for x, v in zip(q, np.quantile(D[:, -1], q))))

    lines.append("\n| tau | docs with NN > tau | docs dropped | tokens dropped (with copies) |\n|---|---|---|---|")
    for tau in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]:
        keep = dedup_mask(I, D, tau)
        lines.append(f"| {tau:.2f} | {(D[:, 0] > tau).mean():.2%} | {(~keep).mean():.2%} | {tok[~keep].sum() / tok.sum():.2%} |")

    ds = lance.dataset(src, version=version)
    bands = [(0.70, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 0.93), (0.93, 0.96), (0.96, 0.98), (0.98, 1.01)]
    pos = dt.src_pos.to_numpy()
    for lo, hi in bands:
        cand = np.flatnonzero((D[:, 0] >= lo) & (D[:, 0] < hi))
        lines.append(f"\n## NN similarity in [{lo:.2f}, {hi:.2f}): {len(cand):,} documents\n")
        pick = rng.choice(cand, size=min(pairs_per_band, len(cand)), replace=False)
        texts = fetch_texts(ds, np.concatenate([pos[pick], pos[I[pick, 0]]]))
        for n, i in enumerate(pick):
            ta, tb = texts[n], texts[n + len(pick)]
            lines.append(f"### {D[i, 0]:.3f}  (doc {i} vs {I[i, 0]})\n")
            lines.append("```text\n" + ta[:600].replace("```", "'''") + "\n```")
            lines.append("```text\n" + tb[:600].replace("```", "'''") + "\n```\n")
    Path(out).write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"-> {out}")

def dedup(doctable, knn_dir, tau, out):
    dt = _doctable(doctable, ["n_tokens", "copies"])
    I, D = load_knn(knn_dir)
    keep = dedup_mask(I, D, tau)
    tok = dt.n_tokens.to_numpy().astype(np.int64) * dt.copies.to_numpy()
    _save(Path(out), keep)
    print(f"tau={tau}: keep {keep.sum():,} of {len(keep):,} documents ({keep.mean():.2%}), "
          f"{tok[keep].sum() / 1e9:.3f}B of {tok.sum() / 1e9:.3f}B tokens with copies -> {out}")

# ---------------------------------------------------------------- step 4: chains

def _walk(active, Il, ntl, cap, deg, used=None):
    """Greedy walk over the `active` documents: start at the lowest-degree unvisited document, step to the most
    similar unvisited neighbour (Il: neighbour lists, -1 = not allowed), start a new chain at a dead end or when
    the next document would pass `cap` tokens. With `used` (a set of pair keys), a pair of documents that was
    adjacent before is never made adjacent again, and the new adjacencies are added to it."""
    N = len(ntl)
    starts = np.flatnonzero(active)
    starts = starts[np.argsort(deg[starts], kind="stable")].tolist()
    vis = (~active).tolist()
    chain_id, chain_pos = [-1] * N, [-1] * N
    cid, p, tok, ptr, cur, prev, n_avoided = -1, 0, 0, 0, -1, -1, 0
    while True:
        if cur < 0:  # dead end: next unvisited document with the lowest degree
            while ptr < len(starts) and vis[starts[ptr]]:
                ptr += 1
            if ptr == len(starts):
                break
            cur = starts[ptr]
            cid, p, tok = cid + 1, 0, 0
        elif tok + ntl[cur] > cap:  # token cap: new chain, same walk
            cid, p, tok = cid + 1, 0, 0
        if p > 0 and used is not None:
            used.add(prev * N + cur if prev < cur else cur * N + prev)
        chain_id[cur], chain_pos[cur] = cid, p
        p, tok, prev = p + 1, tok + ntl[cur], cur
        vis[cur] = True
        nxt = -1
        for j in Il[cur]:
            if j >= 0 and not vis[j]:
                if used is not None and (cur * N + j if cur < j else j * N + cur) in used:
                    n_avoided += 1
                    continue
                nxt = j
                break
        cur = nxt
    return np.array(chain_id, dtype=np.int64), np.array(chain_pos, dtype=np.int32), cid + 1, n_avoided

def build_chains(doctable, knn_dir, keep_path, cap, out, per_pass=False, max_adj=None):
    """Step 4. Default: one walk over all kept documents; repeat passes are derived later (chain restricted to
    copies >= j), so upsampled documents meet the same partners in every pass.
    per_pass: a separate walk for each pass j over the documents with copies >= j, never making two documents
    adjacent again if they were adjacent in an earlier pass. Output then has a pass_ column.
    max_adj: never step to a neighbour more similar than this (keeps near-duplicates apart)."""
    dt = _doctable(doctable, ["n_tokens", "copies"])
    ntok = dt.n_tokens.to_numpy().astype(np.int64)
    copies = dt.copies.to_numpy()
    I, D = load_knn(knn_dir)
    keep = np.load(keep_path)
    N = len(I)
    bad = ~keep[I]
    if max_adj is not None:
        bad |= D > max_adj
        print(f"max_adj={max_adj}: {(D[keep] > max_adj).mean():.2%} of kept documents' neighbour slots disallowed",
              flush=True)
    I = np.where(bad, -1, I)
    del D, bad
    t0 = time.time()
    Il, ntl = I.tolist(), ntok.tolist()
    print(f"neighbour lists ready in {time.time() - t0:.0f}s", flush=True)

    def degree(active):
        ok = (I >= 0) & active[:, None]
        ok &= active[np.where(ok, I, 0)]
        return np.bincount(I[ok], minlength=N)

    if not per_pass:
        chain_id, chain_pos, n_chains, _ = _walk(keep, Il, ntl, cap, degree(keep))
        idx = np.flatnonzero(keep)
        idx = idx[np.lexsort((chain_pos[idx], chain_id[idx]))]
        pq.write_table(pa.table({"doc_idx": idx, "chain_id": chain_id[idx], "chain_pos": chain_pos[idx]}), out)
        print(f"{n_chains:,} chains over {len(idx):,} documents in {time.time() - t0:.0f}s -> {out}")
        return

    used, parts, offset = set(), [], 0
    for j in range(1, int(copies[keep].max()) + 1):
        active = keep & (copies >= j)
        chain_id, chain_pos, n_chains, n_avoided = _walk(active, Il, ntl, cap, degree(active), used)
        idx = np.flatnonzero(active)
        idx = idx[np.lexsort((chain_pos[idx], chain_id[idx]))]
        parts.append(pa.table({"doc_idx": idx, "pass_": np.full(len(idx), j, np.int8),
                               "chain_id": chain_id[idx] + offset, "chain_pos": chain_pos[idx]}))
        offset += n_chains
        print(f"pass {j}: {n_chains:,} chains over {len(idx):,} documents ({len(idx) / n_chains:.2f} per chain); "
              f"{n_avoided:,} steps re-routed to avoid a repeated pair; {time.time() - t0:.0f}s", flush=True)
    pq.write_table(pa.concat_tables(parts), out)
    print(f"{offset:,} rows over {sum(p.num_rows for p in parts):,} document copies, "
          f"{len(used):,} distinct adjacent pairs, none repeated -> {out}")

def inspect_chains(doctable, emb_dir, chains_path, src, version, out, n_show=20, seed=0):
    rng = np.random.default_rng(seed)
    dt = _doctable(doctable, ["src_pos", "n_tokens"])
    ch = _doctable(chains_path)
    E = load_emb(emb_dir)
    doc, cid = ch.doc_idx.to_numpy(), ch.chain_id.to_numpy()
    ntok = dt.n_tokens.to_numpy().astype(np.int64)

    sizes = np.bincount(cid)
    toks = np.bincount(cid, weights=ntok[doc])
    q = [0.1, 0.5, 0.9, 0.99]
    lines = [f"# Chains ({len(sizes):,} chains, {len(doc):,} documents)\n",
             "docs per chain: mean %.2f, " % sizes.mean() + ", ".join(f"p{int(x * 100)}={v:.0f}" for x, v in zip(q, np.quantile(sizes, q))),
             "tokens per chain: mean %.0f, " % toks.mean() + ", ".join(f"p{int(x * 100)}={v:.0f}" for x, v in zip(q, np.quantile(toks, q))),
             f"share of tokens in chains of >= 2 documents: {toks[sizes >= 2].sum() / toks.sum():.2%}",
             f"single-document chains: {(sizes == 1).mean():.2%} of chains"]

    def adjacent(order):
        same = cid[1:] == cid[:-1]
        a, b = order[:-1][same], order[1:][same]
        return (E[a].astype(np.float32) * E[b]).sum(1)
    iclm = adjacent(doc)
    rand = adjacent(rng.permutation(doc))  # same chain lengths, random members
    lines.append(f"adjacent similarity: ICLM mean {iclm.mean():.3f} (median {np.median(iclm):.3f}), "
                 f"random grouping mean {rand.mean():.3f} (median {np.median(rand):.3f})")

    ds = lance.dataset(src, version=version)
    big = np.flatnonzero(sizes >= 3)
    starts = np.concatenate([[0], np.cumsum(sizes)])
    for c in rng.choice(big, size=min(n_show, len(big)), replace=False):
        members = doc[starts[c]:starts[c + 1]]
        texts = fetch_texts(ds, dt.src_pos.to_numpy()[members])
        lines.append(f"\n## chain {c}: {len(members)} documents, {ntok[members].sum():,} tokens\n")
        for m, t in zip(members, texts):
            lines.append(f"- **{m}** ({ntok[m]:,} tok): " + t[:300].replace("\n", " "))
    Path(out).write_text("\n".join(lines))
    print("\n".join(lines[:7]))
    print(f"-> {out}")

# ---------------------------------------------------------------- step 7: random-group control

def build_randgroups(doctable, chains_path, out, block=64, seed=0):
    """Random-grouping control with the ICLM rows' structure: same rows, members swapped for random documents
    of about the same length.

    ICLM row (chain c, pass j) = members of c with copies >= j, in chain order. For each pass j, the documents
    of that pass are sorted by length (random tie-break) and cut into blocks of `block` documents; inside each
    block they are permuted at random, and every ICLM member is replaced by its image. So each control row has
    the same number of documents and nearly the same token length as its ICLM row, while its members are
    unrelated; each document is in exactly one row per pass (never twice in a row, exactly `copies` rows).
    Output columns: doc_idx, pass_, chain_id, chain_pos.
    """
    rng = np.random.default_rng(seed)
    dt = _doctable(doctable, ["n_tokens", "copies"])
    ntok = dt.n_tokens.to_numpy().astype(np.int64)
    copies = dt.copies.to_numpy()
    ch = _doctable(chains_path)
    cdoc, cid = ch.doc_idx.to_numpy(), ch.chain_id.to_numpy()
    per_pass = "pass_" in ch.columns  # chains built per pass (build_chains per_pass=True)
    parts = []
    for j in range(1, int(copies[cdoc].max()) + 1):
        if per_pass:
            m = ch.pass_.to_numpy() == j
            doc, c, pos = cdoc[m], cid[m], ch.chain_pos.to_numpy()[m]
        else:
            m = copies[cdoc] >= j
            doc, c = cdoc[m], cid[m]                      # pass-j rows, members in chain order
            pos = np.arange(len(doc)) - np.searchsorted(c, c)  # rank inside the pass-j row (c is sorted)
        order = np.lexsort((rng.random(len(doc)), ntok[doc]))  # by length, random tie-break
        image = np.empty_like(order)
        for s in range(0, len(order), block):
            blk = order[s:s + block]
            image[blk] = rng.permutation(blk)
        new = doc[image]
        diff = np.abs(np.bincount(c, weights=ntok[new]) - np.bincount(c, weights=ntok[doc]))
        rows = np.unique(c)
        print(f"pass {j}: {len(rows):,} rows, {len(doc):,} documents; row token length differs from ICLM by "
              f"median {np.median(diff[rows]):.0f}, p99 {np.quantile(diff[rows], 0.99):.0f} tokens", flush=True)
        parts.append(pa.table({"doc_idx": new, "pass_": np.full(len(doc), j, np.int8),
                               "chain_id": c, "chain_pos": pos.astype(np.int32)}))
    pq.write_table(pa.concat_tables(parts), out)
    print(f"-> {out}")

# ---------------------------------------------------------------- steps 5-6: write merged rows

MAX_PASS = 16  # row key = group * MAX_PASS + pass

def _groups(groups, keep, copies):
    """Group id and position of every document in every pass: two (n_docs, max_copies) arrays, -1 = no row.

    groups = 'single' (needs keep), a chains parquet (same group in every pass), or a parquet with a pass_ column
    (one group per document and pass, as written by build_randgroups).
    """
    n, jmax = len(copies), int(copies.max())
    G = np.full((n, jmax), -1, dtype=np.int64)
    P = np.zeros((n, jmax), dtype=np.int32)
    if groups == "single":
        if keep is None:
            raise ValueError("--keep is required with --groups single")
        idx = np.flatnonzero(np.load(keep))
        G[idx] = idx[:, None]
    else:
        g = _doctable(groups)
        d = g.doc_idx.to_numpy()
        if "pass_" in g.columns:
            j = g.pass_.to_numpy().astype(np.int64) - 1
            G[d, j], P[d, j] = g.chain_id.to_numpy(), g.chain_pos.to_numpy()
        else:
            G[d] = g.chain_id.to_numpy()[:, None]
            P[d] = g.chain_pos.to_numpy()[:, None]
    G[np.arange(jmax)[None, :] >= copies[:, None]] = -1  # pass j exists only for documents with copies >= j
    inset = G[:, 0] >= 0
    if ((G >= 0).sum(1) != np.where(inset, copies, 0)).any():
        raise ValueError(f"{groups}: some document is missing from a pass it should be in")
    return G, P

def _bucket_of(key, n_buckets, seed):
    with np.errstate(over="ignore"):
        x = (key.astype(np.uint64) + np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
    return ((x >> np.uint64(33)) % np.uint64(n_buckets)).astype(np.int64)

_SCATTER_SCHEMA = pa.schema([("key", pa.int64()), ("pos", pa.int32()), ("id", pa.string()),
                             ("n_tokens", pa.int32()), ("subset", pa.string()), ("text", pa.large_string())])

def _scatter(job, src, version, tmp, n_buckets, seed):
    """Pass 1 (worker): read one source fragment's documents, expand repeats into rows, send rows to buckets."""
    fid, local, G, P, copies, ids, ntok = job
    out = [tmp / f"b{b:04d}" / f"f{fid:05d}.arrow" for b in range(n_buckets)]
    if all(p.exists() for p in out):
        return fid, 0
    frag = lance.dataset(src, version=version).get_fragment(fid)
    # read in pieces: a fragment's selected texts can exceed the 2 GB limit of 32-bit string offsets
    pieces = [frag.take(local[s:s + 32_768], columns=["text", "subset"]) for s in range(0, len(local), 32_768)]
    t = pa.concat_tables([p.set_column(p.schema.get_field_index("text"), "text", p["text"].cast(pa.large_string()))
                          for p in pieces])
    rep = np.repeat(np.arange(len(local)), copies)                       # pass j = 1..copies of each document
    j = np.arange(len(rep)) - np.repeat(np.cumsum(copies) - copies, copies) + 1
    key = G[rep, j - 1] * MAX_PASS + j
    rows = pa.table({
        "key": key, "pos": P[rep, j - 1], "id": pa.array(ids).take(rep), "n_tokens": ntok[rep],
        "subset": t["subset"].take(rep), "text": t["text"].cast(pa.large_string()).take(rep),
    }, schema=_SCATTER_SCHEMA)
    bucket = _bucket_of(key, n_buckets, seed)
    order = np.argsort(bucket, kind="stable")
    bounds = np.searchsorted(bucket[order], np.arange(n_buckets + 1))
    rows = rows.take(order)
    opts = pa.ipc.IpcWriteOptions(compression="lz4")
    for b in range(n_buckets):
        tmp_path = out[b].with_suffix(".part")
        with pa.OSFile(str(tmp_path), "wb") as f, pa.ipc.new_file(f, _SCATTER_SCHEMA, options=opts) as w:
            w.write_table(rows.slice(bounds[b], bounds[b + 1] - bounds[b]))
        os.replace(tmp_path, out[b])
    return fid, len(rep)

OUT_SCHEMA = pa.schema([
    ("text", pa.large_string()), ("n_tokens", pa.int32()), ("n_docs", pa.int16()), ("chain_id", pa.int64()),
    ("pass_", pa.int8()), ("ids", pa.list_(pa.string())), ("doc_n_tokens", pa.list_(pa.int32())),
    ("subset", pa.list_(pa.string())),
])

def _gather(b, tmp, dst, sep, rows_per_file, seed):
    """Pass 2 (worker): build the rows of one bucket, shuffle them, write uncommitted fragments."""
    from lance.fragment import write_fragments
    files = sorted((tmp / f"b{b:04d}").glob("f*.arrow"))
    t = pa.concat_tables([pa.ipc.open_file(pa.memory_map(str(f))).read_all() for f in files])
    key, pos = t["key"].to_numpy(), t["pos"].to_numpy()
    order = np.lexsort((pos, key))
    t = t.take(order)
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    offsets = np.r_[starts, len(key)].astype(np.int32)
    texts = t["text"].to_pylist()
    joined = [sep.join(texts[s:e]) for s, e in zip(offsets[:-1], offsets[1:])]
    del texts
    ntok = t["n_tokens"].to_numpy().astype(np.int64)
    rowkey = key[starts]
    rows = pa.table({
        "text": pa.array(joined, type=pa.large_string()),
        "n_tokens": np.add.reduceat(ntok, starts).astype(np.int32),
        "n_docs": np.diff(offsets).astype(np.int16),
        "chain_id": rowkey // MAX_PASS,
        "pass_": (rowkey % MAX_PASS).astype(np.int8),
        "ids": pa.ListArray.from_arrays(pa.array(offsets), t["id"].combine_chunks()),
        "doc_n_tokens": pa.ListArray.from_arrays(pa.array(offsets), t["n_tokens"].combine_chunks()),
        "subset": pa.ListArray.from_arrays(pa.array(offsets), t["subset"].combine_chunks()),
    }, schema=OUT_SCHEMA)
    del joined, t
    rows = rows.take(np.random.default_rng([seed, b]).permutation(rows.num_rows))
    batches = rows.to_batches(max_chunksize=4096)
    frags = write_fragments(pa.RecordBatchReader.from_batches(OUT_SCHEMA, batches), dst, schema=OUT_SCHEMA,
                            max_rows_per_file=rows_per_file, data_storage_version="2.2")
    return b, rows.num_rows, int(rows["n_tokens"].to_numpy().sum()), frags

def write_dataset(doctable, groups, keep, src, version, dst, tmp, n_buckets=128, scatter_workers=24,
                  gather_workers=16, rows_per_file=32_768, seed=0, tokenizer="meta-llama/Llama-3.2-1B"):
    """Steps 5-6: one row per (group, pass j) with the group's members of copies >= j, joined with EOS+BOS."""
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    sep = tok.eos_token + tok.bos_token
    if Path(dst).exists():
        raise FileExistsError(dst)
    dt = _doctable(doctable, ["src_pos", "id", "copies", "n_tokens"])
    copies = dt.copies.to_numpy().astype(np.int64)
    G, P = _groups(groups, keep, copies)
    inset = G[:, 0] >= 0
    expect_rows_docs = int(copies[inset].sum())
    expect_tok = int((dt.n_tokens.to_numpy().astype(np.int64) * copies)[inset].sum())
    print(f"{inset.sum():,} documents, {expect_rows_docs:,} document copies, {expect_tok / 1e9:.3f}B tokens "
          f"-> {dst}", flush=True)

    ds = lance.dataset(src, version=version)
    frags = ds.get_fragments()
    counts = np.array([f.count_rows() for f in frags])
    bounds = np.r_[0, np.cumsum(counts)]
    src_pos = dt.src_pos.to_numpy()
    ids = dt.id.to_numpy()
    ntok = dt.n_tokens.to_numpy()
    tmp = Path(tmp)
    for b in range(n_buckets):
        (tmp / f"b{b:04d}").mkdir(parents=True, exist_ok=True)

    def jobs():
        for f, lo, hi in zip(frags, bounds[:-1], bounds[1:]):
            a, z = np.searchsorted(src_pos, [lo, hi])
            d = np.arange(a, z)[inset[a:z]]
            if len(d):
                yield (f.fragment_id, src_pos[d] - lo, G[d], P[d], copies[d], ids[d].tolist(), ntok[d])

    t0, n = time.time(), 0
    with ProcessPoolExecutor(scatter_workers) as ex:
        for i, (fid, k) in enumerate(ex.map(partial(_scatter, src=src, version=version, tmp=tmp,
                                                    n_buckets=n_buckets, seed=seed), jobs())):
            n += k
            print(f"scatter: fragment {fid} ({i + 1}/{len(frags)}), {n:,} document copies, "
                  f"{time.time() - t0:.0f}s", flush=True)

    t0, all_frags, n_rows, n_tok = time.time(), [], 0, 0
    with ProcessPoolExecutor(gather_workers) as ex:
        for b, r, k, fr in ex.map(partial(_gather, tmp=tmp, dst=dst, sep=sep, rows_per_file=rows_per_file,
                                          seed=seed), range(n_buckets)):
            all_frags += fr
            n_rows, n_tok = n_rows + r, n_tok + k
            print(f"gather: bucket {b + 1}/{n_buckets}, {n_rows:,} rows, {n_tok / 1e9:.3f}B tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
    if n_tok != expect_tok:
        raise RuntimeError(f"token total {n_tok} != expected {expect_tok}; not committing")
    lance.LanceDataset.commit(dst, lance.LanceOperation.Overwrite(OUT_SCHEMA, all_frags), read_version=0)
    out = lance.dataset(dst)
    print(f"committed {dst} v{out.version}: {out.count_rows():,} rows, {len(out.get_fragments())} fragments, "
          f"{n_tok / 1e9:.3f}B tokens", flush=True)
    shutil.rmtree(tmp)

# ---------------------------------------------------------------- checks

def check_dataset(dst, doctable, groups, keep, n_tokenize=300, seed=0, tokenizer="meta-llama/Llama-3.2-1B"):
    """Each id appears exactly `copies` times, never twice in a row; token totals and n_tokens are exact."""
    from transformers import AutoTokenizer
    ds = lance.dataset(dst)
    dt = _doctable(doctable, ["id", "copies", "n_tokens"])
    G, _ = _groups(groups, keep, dt.copies.to_numpy().astype(np.int64))
    inset = G[:, 0] >= 0
    t = ds.to_table(columns=["n_tokens", "n_docs", "ids", "doc_n_tokens", "pass_"])
    ok = True

    def report(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}", flush=True)

    ntok = t["n_tokens"].to_numpy().astype(np.int64)
    expect = int((dt.n_tokens.to_numpy().astype(np.int64) * dt.copies.to_numpy())[inset].sum())
    report("token total", ntok.sum() == expect, f"{ntok.sum():,} vs {expect:,}")
    dnt = pc.list_flatten(t["ids"])  # per chunk (a combined array would pass 2 GB); keeps row order
    lens = pc.list_value_length(t["ids"]).to_numpy()
    report("n_docs = len(ids)", (lens == t["n_docs"].to_numpy()).all())
    sums = np.add.reduceat(pc.list_flatten(t["doc_n_tokens"]).to_numpy().astype(np.int64), np.r_[0, np.cumsum(lens)[:-1]])
    report("n_tokens = sum(doc_n_tokens)", (sums == ntok).all())

    row = np.repeat(np.arange(len(lens)), lens)
    flat = pd.DataFrame({"row": row, "id": dnt.to_numpy(zero_copy_only=False)})
    report("no id twice in one row", not flat.duplicated().any())
    got = flat.id.value_counts()
    want = pd.Series(dt.copies.to_numpy()[inset].astype(np.int64), index=dt.id.to_numpy()[inset])
    got = got.reindex(want.index).fillna(0).astype(np.int64)
    report("every document appears exactly `copies` times", (got == want).all(),
           f"({(got != want).sum():,} mismatches, {len(flat.id.unique()) - len(want):,} unexpected ids)")

    tok = AutoTokenizer.from_pretrained(tokenizer)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(ds.count_rows(), size=min(n_tokenize, ds.count_rows()), replace=False))
    sample = ds.take(idx, columns=["text", "n_tokens", "n_docs"]).to_pylist()
    bad = [r for r in sample if len(tok(r["text"])["input_ids"]) + 1 != r["n_tokens"]]
    report(f"tokenized length + EOS = n_tokens on {len(sample)} rows", not bad, f"({len(bad)} differ)")

    q = [0.1, 0.5, 0.9, 0.99]
    print(f"rows {len(ntok):,}; tokens per row mean {ntok.mean():.0f}, "
          + ", ".join(f"p{int(x * 100)}={v:.0f}" for x, v in zip(q, np.quantile(ntok, q)))
          + f"; docs per row mean {lens.mean():.2f}; rows per pass "
          + str(dict(zip(*np.unique(t["pass_"].to_numpy(), return_counts=True)))), flush=True)
    if not ok:
        raise SystemExit(1)

# ---------------------------------------------------------------- CLI

def get_args():
    p = ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctable"); s.add_argument("--src", required=True); s.add_argument("--version", type=int, required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("sample"); s.add_argument("--doctable", required=True); s.add_argument("--n", type=int, required=True); s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("embed")
    for a in ("--doctable", "--src", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--version", type=int, required=True); s.add_argument("--model", default="facebook/contriever")
    s.add_argument("--chunk", type=int, default=65_536); s.add_argument("--batch-tokens", type=int, default=262_144)
    s = sub.add_parser("knn"); s.add_argument("--emb", required=True); s.add_argument("--out", required=True); s.add_argument("--k", type=int, default=10)
    s.add_argument("--qblock", type=int, default=8192); s.add_argument("--dchunk", type=int, default=1 << 20)
    s = sub.add_parser("simstats")
    for a in ("--doctable", "--emb", "--knn", "--src", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--version", type=int, required=True)
    s = sub.add_parser("dedup"); s.add_argument("--doctable", required=True); s.add_argument("--knn", required=True); s.add_argument("--tau", type=float, required=True); s.add_argument("--out", required=True)
    s = sub.add_parser("chains")
    for a in ("--doctable", "--knn", "--keep", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--cap", type=int, default=32_768)
    s.add_argument("--per-pass", action="store_true", help="separate walk per repeat pass, no pair adjacent twice")
    s.add_argument("--max-adj", type=float, help="never step to a neighbour more similar than this")
    s = sub.add_parser("inspect")
    for a in ("--doctable", "--emb", "--chains", "--src", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--version", type=int, required=True)
    s = sub.add_parser("randgroup")
    for a in ("--doctable", "--chains", "--out"):
        s.add_argument(a, required=True)
    s.add_argument("--block", type=int, default=64); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("write", help="steps 5-6: --groups is a chains/randgroup parquet, or 'single' (needs --keep)")
    for a in ("--doctable", "--groups", "--src", "--out", "--tmp"):
        s.add_argument(a, required=True)
    s.add_argument("--keep"); s.add_argument("--version", type=int, required=True)
    s.add_argument("--buckets", type=int, default=128); s.add_argument("--scatter-workers", type=int, default=24)
    s.add_argument("--gather-workers", type=int, default=16); s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("check")
    for a in ("--dataset", "--doctable", "--groups"):
        s.add_argument(a, required=True)
    s.add_argument("--keep")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "doctable":
        build_doctable(args.src, args.version, args.out)
    elif args.cmd == "sample":
        sample(args.doctable, args.n, args.out, args.seed)
    elif args.cmd == "embed":
        embed(args.doctable, args.src, args.version, args.out, args.model, args.chunk, args.batch_tokens)
    elif args.cmd == "knn":
        knn(args.emb, args.out, args.k, args.qblock, args.dchunk)
    elif args.cmd == "simstats":
        simstats(args.doctable, args.emb, args.knn, args.src, args.version, args.out)
    elif args.cmd == "dedup":
        dedup(args.doctable, args.knn, args.tau, args.out)
    elif args.cmd == "chains":
        build_chains(args.doctable, args.knn, args.keep, args.cap, args.out, args.per_pass, args.max_adj)
    elif args.cmd == "inspect":
        inspect_chains(args.doctable, args.emb, args.chains, args.src, args.version, args.out)
    elif args.cmd == "randgroup":
        build_randgroups(args.doctable, args.chains, args.out, args.block, args.seed)
    elif args.cmd == "write":
        write_dataset(args.doctable, args.groups, args.keep, args.src, args.version, args.out, args.tmp,
                      args.buckets, args.scatter_workers, args.gather_workers, seed=args.seed)
    elif args.cmd == "check":
        check_dataset(args.dataset, args.doctable, args.groups, args.keep)
