"""Generate the report figures (PDF) from results/. Usage: python scripts/make_figures.py"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
FIG = Path("report/figures")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.bbox": "tight"})
load = lambda p: json.load(open(p))  # noqa: E731


def training_curves():
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    for name, label in [("qwen3-4b", "Qwen3-4B"), ("qwen3-1.7b", "Qwen3-1.7B")]:
        h = load(R / f"{name}-train_log.json")["val_history"]
        ep = [r["epoch"] for r in h]
        axs[0].plot(ep, [r["val_accuracy"] for r in h], "o-", label=f"{label} acc")
        axs[0].plot(ep, [r["val_macro_f1"] for r in h], "s--", label=f"{label} macro-F1")
        axs[1].plot(ep, [r["val_evidence_micro_f1"] for r in h], "^-", label=label)
    axs[0].set(xlabel="epoch", ylabel="validation score", title="Label accuracy and macro-F1")
    axs[1].set(xlabel="epoch", ylabel="evidence micro-F1", title="Evidence retrieval")
    for a in axs:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a.set_xticks([1, 2, 3])
    fig.savefig(FIG / "training_curves.pdf")
    plt.close(fig)


def kv_cache():
    b = load(R / "bench_qwen3-4b.json")["kv_cache"]
    outs = sorted({r["output_tokens"] for r in b})
    fig, ax = plt.subplots(figsize=(4.2, 3))
    for cache, mark, ls in [(True, "o", "-"), (False, "x", "--")]:
        for n_out in outs:
            rows = [r for r in b if r["cache"] == cache and r["output_tokens"] == n_out]
            rows.sort(key=lambda r: r["input_tokens"])
            ax.plot([r["input_tokens"] for r in rows], [r["decode_tok_per_s"] for r in rows],
                    marker=mark, linestyle=ls, label=f"{'cached' if cache else 'uncached'}, {n_out} out")
    ax.set(xlabel="prompt length (tokens)", ylabel="decode throughput (tokens/s)",
           title="KV cache vs recompute (Qwen3-4B)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "kv_cache.pdf")
    plt.close(fig)


def attention():
    b = load(R / "bench_qwen3-4b.json")["attention"]["standalone"]
    variants = ["reference", "tiled_64", "sdpa"]
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8))
    for v in variants:
        rows = sorted([r for r in b if r["variant"] == v and "ms_mean" in r], key=lambda r: r["seq_len"])
        axs[0].plot([r["seq_len"] for r in rows], [r["ms_mean"] for r in rows], "o-", label=v)
        err = [(r["seq_len"], r["max_abs_err"]) for r in rows if r.get("max_abs_err") is not None and v != "reference"]
        if err:
            axs[1].plot([e[0] for e in err], [e[1] for e in err], "o-", label=v)
    axs[0].set(xlabel="sequence length", ylabel="time (ms)", title="Attention latency", xscale="log", yscale="log")
    axs[1].set(xlabel="sequence length", ylabel="max abs error vs fp32 reference", title="Numerical agreement", xscale="log")
    for a in axs:
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.savefig(FIG / "attention.pdf")
    plt.close(fig)


def speculative():
    b = [r for r in load(R / "bench_qwen3-4b.json")["speculative"] if r["task"] == "explanation"]
    ks = sorted({r["k"] for r in b})
    acc = [np.mean([r["acceptance_rate"] for r in b if r["k"] == k]) for k in ks]
    spd = [np.mean([r["speedup"] for r in b if r["k"] == k]) for k in ks]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.plot(ks, acc, "o-", label="draft acceptance rate")
    ax.plot(ks, spd, "s-", label="speedup vs greedy")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set(xlabel="draft length k", ylabel="rate / speedup", title="Speculative decoding (explanation step)")
    ax.set_xticks(ks)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.savefig(FIG / "speculative.pdf")
    plt.close(fig)


def confusion():
    for name, title in [("qwen3-4b-lora", "Qwen3-4B fine-tuned"), ("qwen3-4b-base", "Qwen3-4B base")]:
        c = load(R / name / "test_metrics.json")["overall"]["labels"]["confusion"]
        labels, mat = c["labels"], np.array(c["matrix"])
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(labels)), [l.replace(" ", "\n") for l in labels], fontsize=6)
        ax.set_yticks(range(len(labels)), labels, fontsize=6)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set(xlabel="predicted", ylabel="expert (true)", title=title)
        fig.colorbar(im, fraction=0.046)
        fig.savefig(FIG / f"confusion_{name}.pdf")
        plt.close(fig)


training_curves()
kv_cache()
attention()
speculative()
confusion()
print("figures written to", FIG, ":", sorted(p.name for p in FIG.glob("*.pdf")))
