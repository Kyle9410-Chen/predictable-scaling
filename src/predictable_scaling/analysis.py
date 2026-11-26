from predictable_scaling.dataset_utils import read
import numpy as np
import matplotlib.pyplot as plt

COLUMNS = ["mtld", "flesch_reading_ease", "n_tokens", "fineweb_edu", "fineweb_edu_int"]

def load(src):
    t = read(src, columns=COLUMNS, limit=None, max_rows=None, fmt="arrow")
    out = {}
    for c in COLUMNS:
        x = t[c].to_numpy()
        out[c] = x[np.isfinite(x)]
    return out

RANGE_PCT = {"mtld": (0, 99.5), "flesch_reading_ease": (5, 99.5), "n_tokens": (0, 80),
             "fineweb_edu": (0, 100), "fineweb_edu_int": (0, 100)}

def plot(data, out):
    fig, axes = plt.subplots(1, len(data), figsize=(13, 4.5))
    for ax, (name, x) in zip(axes, data.items()):
        low, high = np.percentile(x, RANGE_PCT[name])
        ax.hist(x, bins=100, range=(low, high), color='#2a78d6')

        shown = ((x >= low) & (x <= high)).mean()
        ax.set_title(f"{name}  ({1 - shown:.1%} outside range not shown)", loc="left")
        
        med = np.median(x)
        ax.set_xlabel(name)
        ax.axvline(med, color='black')
        ax.text(med, ax.get_ylim()[1] * 0.95, f" median {med:.1f}", va="top")

    fig.tight_layout()
    plt.savefig(out, dpi=150)
    print("saved", out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src")
    p.add_argument("--out")
    args = p.parse_args()

    plot(load(args.src), args.out)