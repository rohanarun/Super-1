#!/usr/bin/env python3
"""
Figures derived only from verified artifacts.

Inputs:
  * auto-research/results/synthetic-rotation-radial-20261016-v1/execution/<seed>-<arm>.json
    (12 rows: confirmation CE, selection CE, CPU; each row carries its own
    checkpoint_sha256 and freeze_sha256_before_confirmation)
  * auto-research/results/allocation-ablation-20261017-v1/PARAMETER_ALLOCATION.json

Nothing here trains, tunes or selects. Every figure states its own provenance and
carries the negative result forward rather than hiding it. Run:

    python3 make_graphs.py
"""

import json
import math
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RADIAL = os.path.join(REPO, "auto-research", "results",
                      "synthetic-rotation-radial-20261016-v1", "execution")
ALLOC = os.path.join(REPO, "auto-research", "results",
                     "allocation-ablation-20261017-v1", "PARAMETER_ALLOCATION.json")
OUT = os.path.join(REPO, "auto-research", "graphs",
                   "allocation-ablation-20261017-v1")

SEEDS = (73, 191, 307)
ARMS = ("native", "rotation", "radial", "pinned_nanogpt")
LABEL = {"native": "hierarchy native", "rotation": "rotation",
         "radial": "radial control", "pinned_nanogpt": "nanoGPT"}

# ln(8): the readouts are uniform over eight classes, so this is the floor for the
# primary readout metric. Arms BELOW it are worse than a uniform guess.
LN8 = math.log(8)


def load():
    rows = {}
    for seed in SEEDS:
        for arm in ARMS:
            path = os.path.join(RADIAL, f"{seed}-{arm}.json")
            if not os.path.exists(path):
                continue
            with open(path) as fh:
                rows[(seed, arm)] = json.load(fh)
    return rows


def fig_nanogpt_wins(rows, path):
    arms = [a for a in ARMS if any(k[1] == a for k in rows)]
    means, lows, highs = [], [], []
    for arm in arms:
        vals = [rows[(s, arm)]["confirmation"]["ce_nats_per_target"]
                for s in SEEDS if (s, arm) in rows]
        means.append(statistics.mean(vals))
        if len(vals) > 1:
            half = (max(vals) - min(vals)) / 2.0
            lows.append(half)
            highs.append(half)
        else:
            lows.append(0)
            highs.append(0)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ypos = range(len(arms))
    colors = ["#c0392b" if a == "pinned_nanogpt" else "#7f8c8d" for a in arms]
    ax.errorbar(means, list(ypos), xerr=[lows, highs], fmt="o", ms=9,
                capsize=5, lw=2, color="#2c3e50", ecolor="#95a5a6")
    for y, arm, m in zip(ypos, arms, means):
        ax.scatter([m], [y], s=150, zorder=3, color=colors[arms.index(arm)],
                   edgecolor="black", linewidth=0.6)
    ax.axvline(LN8, color="#2980b9", ls="--", lw=1.6,
               label=f"ln(8) = {LN8:.3f}  (uniform-guess floor)")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([LABEL[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("confirmation readout CE (nats/target, lower is better)")
    ax.set_title("nanoGPT beats every hierarchy arm on the 2026-10-16 synthetic task\n"
                 "3 paired seeds; bars are observed min-max", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_selection_vs_confirmation(rows, path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    markers = {"native": "o", "rotation": "s", "radial": "^",
               "pinned_nanogpt": "D"}
    colors = {"native": "#7f8c8d", "rotation": "#e67e22",
              "radial": "#27ae60", "pinned_nanogpt": "#c0392b"}
    for arm in ARMS:
        xs, ys = [], []
        for s in SEEDS:
            if (s, arm) not in rows:
                continue
            r = rows[(s, arm)]
            xs.append(r["initial_selection"]["ce_nats_per_target"])
            ys.append(r["final_selection"]["ce_nats_per_target"])
        if not xs:
            continue
        ax.plot(xs, ys, markers[arm], ms=9, ls="none", color=colors[arm],
                label=LABEL[arm], alpha=0.85)
    lim = [2.0, 4.2]
    ax.plot(lim, lim, color="#bdc3c7", lw=1, ls="-", zorder=0)
    ax.annotate("no learning", xy=(3.1, 3.1), xytext=(2.25, 3.55),
                fontsize=9, color="#7f8c8d",
                arrowprops=dict(arrowstyle="->", color="#bdc3c7"))
    ax.set_xlabel("initial selection readout CE")
    ax.set_ylabel("final selection readout CE")
    ax.set_title("Hierarchy readout CE gets WORSE over training\n"
                 "7:1 command-to-readout loss weighting pulls it off the readout",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_allocation(path):
    with open(ALLOC) as fh:
        alloc = json.load(fh)
    hier = alloc["hierarchy_by_width_at_qwen_vocab"]
    nano = alloc["nanogpt_by_shape"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    names = ["hier d=688"] + [f"nanoGPT {n['width']}/{n['layers']}L" for n in nano]
    embeds = [hier[4]["embedding"]] + [n["embedding"] for n in nano]
    bodies = [hier[4]["body"]] + [n["body"] for n in nano]
    x = range(len(names))
    ax1.bar(x, embeds, label="embedding", color="#95a5a6")
    ax1.bar(x, bodies, bottom=embeds, label="body", color="#c0392b")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("parameters")
    ax1.set_title("At matched total, the baseline gets ~2x the body\n"
                  "(V=151936)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    lever = alloc["allocation_lever_split_embed_width"]
    xs = [r["embed_width"] for r in lever]
    ys = [r["body_fraction_pct"] for r in lever]
    ax2.plot(xs, ys, "o-", color="#2c3e50", lw=2)
    for r in lever:
        ax2.annotate(f"{r['total']/1e6:.0f}M", (r["embed_width"],
                     r["body_fraction_pct"]),
                     textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax2.set_xlabel("input embedding width d_embed")
    ax2.set_ylabel("body share of total params (%)")
    ax2.set_title("Cheapest declared lever: shrink the tied embedding\n"
                  "(analytic counts only, no training)", fontsize=11)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()
    if len(rows) < 12:
        raise SystemExit(f"expected 12 verified arm/seed rows, found {len(rows)}")
    fig_nanogpt_wins(rows, os.path.join(OUT, "fig1_nanogpt_wins_synthetic.png"))
    fig_selection_vs_confirmation(rows, os.path.join(OUT, "fig2_selection_regression.png"))
    fig_allocation(os.path.join(OUT, "fig3_parameter_allocation.png"))
    print("wrote 3 figures to", OUT)


if __name__ == "__main__":
    main()