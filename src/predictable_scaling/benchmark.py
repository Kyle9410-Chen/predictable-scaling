"""Benchmarks of the ICLM study (docs/iclm_results.md §3). Subcommands:

    evaluate         run lm-eval tasks on one checkpoint (float32, per-sample logs)
                     -> OUT/<checkpoint dir name>/<task>_<shots>shot/                              (GPU)
    language-modeling  bits per byte (wikitext, c4, pile_10k), LAMBADA accuracy, BLiMP          (CPU)
    multiple-choice    probability metrics (DataDecide-style), few-shot gain                     (CPU)

The two analyses compare the models evaluated into OUT (every pair, or the pairs given with --pair) with a
paired bootstrap: each model is evaluated once, items are resampled with replacement 2,000 times, 95%
percentile interval; `*` = interval excludes 0.

lm-eval (lm-evaluation-harness) runs from its own venv, called as a subprocess (--lm-eval): the project venv is
Python 3.14, which lm-eval does not support. Install it once, e.g. (the versions used for the reported results):
    uv venv --python 3.12 /work/$USER/venvs/lmeval
    uv pip install --python /work/$USER/venvs/lmeval torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python /work/$USER/venvs/lmeval "lm_eval[hf]==0.4.13" transformers==5.20.1
"""
import glob, itertools, json, math, os, re, subprocess
from argparse import ArgumentParser

import numpy as np

LM_TASKS = ["wikitext", "c4", "pile_10k", "lambada_openai", "lambada_standard", "blimp"]
MC_TASKS = ["arc_easy:0", "arc_challenge:0", "hellaswag:0", "piqa:0", "sciq:0", "winogrande:0", "openbookqa:0",
            "copa:0", "sst2:0", "sst2:4", "sst2:16", "boolq:0", "boolq:4", "rte:0", "rte:4", "rte:16", "race:0",
            "arc_easy:5", "piqa:5", "hellaswag:5"]
LIMIT = {"c4": 2000, "pile_10k": 2000, **{t.split(":")[0]: 2000 for t in MC_TASKS}}
rng = np.random.default_rng(0)


def boot(stat, n, b=2000):
    """stat(idx) -> values for index arrays of shape (k, n). Returns point estimate and 95% interval."""
    s = stat(rng.integers(0, n, size=(b, n)))
    return stat(np.arange(n)[None])[0], *np.quantile(s, [0.025, 0.975])


def fmt(p, lo, hi, d=4):
    return f"{p:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{'*' if lo > 0 or hi < 0 else ''}"


def samples(out, model, run):
    """(sub-task, doc_id) -> per-sample record of one lm-eval run (file names carry a timestamp; strip it)."""
    recs = {}
    for f in sorted(glob.glob(f"{out}/{model}/{run}/**/samples_*.jsonl", recursive=True)):
        sub = re.sub(r"_\d{4}-\d\d-\d\dT.*", "", os.path.basename(f))
        for line in open(f):
            d = json.loads(line)
            recs[(sub, d["doc_id"])] = d
    return recs


def models_and_pairs(args):
    models = sorted(d for d in os.listdir(args.out) if os.path.isdir(f"{args.out}/{d}"))
    return models, [tuple(p) for p in args.pair] or list(itertools.combinations(models, 2))


def evaluate(args):
    name = os.path.basename(os.path.normpath(args.model))
    for job in [f"{t}:0" for t in LM_TASKS] + MC_TASKS:
        task, shots = job.split(":")
        o = f"{args.out}/{name}/{task}_{shots}shot"
        if glob.glob(f"{o}/**/results_*.json", recursive=True):
            continue
        cmd = [args.lm_eval, "--model", "hf", "--tasks", task, "--num_fewshot", shots, "--batch_size", "8",
               "--model_args", f"pretrained={args.model},dtype=float32,add_bos_token=True,max_length=8192",
               "--log_samples", "--output_path", o] + (["--limit", str(LIMIT[task])] if task in LIMIT else [])
        ok = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
        print(f"[{name}] {task} {shots}-shot {'done' if ok else 'FAILED'}", flush=True)


def language_modeling(args):
    models, pairs = models_and_pairs(args)
    for task in ("wikitext", "c4", "pile_10k"):
        def arr(m):
            r = samples(args.out, m, f"{task}_0shot")
            k = sorted(r)
            return np.array([float(r[i]["bits_per_byte"][0]) for i in k]), np.array([float(r[i]["bits_per_byte"][1]) for i in k])
        data = {m: arr(m) for m in models}
        print(f"\n{task} bits per byte")
        for m, (ll, nb) in data.items():
            print(f"  {m}: {-ll.sum() / (nb.sum() * math.log(2)):.4f}")
        for a, b in pairs:
            (la, nb), (lb, _) = data[a], data[b]
            print(f"  {a} - {b}: " + fmt(*boot(lambda i: (lb[i].sum(1) - la[i].sum(1)) / (nb[i].sum(1) * math.log(2)), len(la))))
    for task in ("lambada_openai", "lambada_standard"):
        data = {m: samples(args.out, m, f"{task}_0shot") for m in models}
        print(f"\n{task} accuracy")
        for m, r in data.items():
            print(f"  {m}: {np.mean([float(d['acc']) for d in r.values()]):.4f}")
        for a, b in pairs:
            k = sorted(data[a])
            x = np.array([float(data[a][i]["acc"]) - float(data[b][i]["acc"]) for i in k])
            print(f"  {a} - {b}: " + fmt(*boot(lambda i: x[i].mean(1), len(x))))
    print("\nblimp accuracy")
    for m in models:
        r = glob.glob(f"{args.out}/{m}/blimp_0shot/**/results_*.json", recursive=True)
        print(f"  {m}: {json.load(open(r[0]))['results']['blimp']['acc,none']:.4f}" if r else f"  {m}: missing")


def option_metrics(d):
    """Per item: l_i = option log-likelihood, q_i = exp(l_i / chars_i) (acc_norm's length normalisation)."""
    ll = np.array([float(r[0]) for r in d["filtered_resps"]])
    opts = [a["arg_1"] for a in d["arguments"].values()]
    ch = np.array([max(len(o), 1) for o in opts], dtype=float)
    t = str(d["target"]).strip()
    c = int(t) if t.lstrip("-").isdigit() else [o.strip() for o in opts].index(t)  # COPA logs the option text
    q = np.exp(ll / ch)
    return {"acc": float(ll.argmax() == c), "acc_norm": float((ll / ch).argmax() == c), "correct_prob": q[c],
            "total_prob": q.sum(), "norm_prob": q[c] / q.sum(), "margin": q[c] - np.delete(q, c).max()}


def multiple_choice(args):
    models, pairs = models_and_pairs(args)
    runs = [f"{j.split(':')[0]}_{j.split(':')[1]}shot" for j in MC_TASKS]
    data = {(m, r): {k: option_metrics(d) for k, d in samples(args.out, m, r).items()} for m in models for r in runs}
    keys = ["acc", "acc_norm", "correct_prob", "total_prob", "norm_prob", "margin"]
    print("run model " + " ".join(keys))
    for r in runs:
        for m in models:
            if data[(m, r)]:
                print(r, m, " ".join(f"{np.mean([e[k] for e in data[(m, r)].values()]):.4f}" for k in keys))
    print("\npaired A - B: norm_prob, margin")
    for r in runs:
        for a, b in pairs:
            ids = sorted(set(data[(a, r)]) & set(data[(b, r)]))
            cols = [fmt(*boot(lambda i, x=np.array([data[(a, r)][j][k] - data[(b, r)][j][k] for j in ids]): x[i].mean(1), len(ids)))
                    for k in ("norm_prob", "margin")]
            print(r, f"{a} - {b}", *cols)
    print("\nfew-shot gain: k-shot minus 0-shot norm_prob, same items")
    for r in runs:
        task, shots = r.rsplit("_", 1)
        if shots != "0shot":
            for m in models:
                d0, dk = data[(m, f"{task}_0shot")], data[(m, r)]
                ids = sorted(set(d0) & set(dk))
                x = np.array([dk[i]["norm_prob"] - d0[i]["norm_prob"] for i in ids])
                print(r, m, fmt(*boot(lambda i: x[i].mean(1), len(x))))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["evaluate", "language-modeling", "multiple-choice"])
    p.add_argument("--out", required=True, help="results folder")
    p.add_argument("--model", help="evaluate: checkpoint path")
    p.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                   help="analyses: compare A - B (checkpoint dir names); repeatable; default every pair")
    p.add_argument("--lm-eval", default=f"/work/{os.environ.get('USER')}/venvs/lmeval/bin/lm_eval",
                   help="evaluate: path to the lm_eval executable of the separate lm-eval venv (see module docstring)")
    args = p.parse_args()
    {"evaluate": evaluate, "language-modeling": language_modeling, "multiple-choice": multiple_choice}[args.cmd](args)
