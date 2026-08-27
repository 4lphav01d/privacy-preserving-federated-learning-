#!/usr/bin/env python
"""Chapter 5 analysis: extract grid results from results.jsonl, build tables + figures.

Run after the campaign completes:  ./.venv/bin/python analyse.py
Outputs:
  - results/summary_table.csv          (RQ1/RQ2 accuracy-vs-epsilon, both datasets)
  - results/compression_table.csv      (RQ3 communication + accuracy price)
  - results/inversion_table.csv        (RQ4 L2/SSIM)
  - chapter_4_5_6/images/Figure_5.{1,2,3}.png
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(ROOT, "..", "chapter_4_5_6", "images")

EPS_ORDER = [None, 8, 4, 2, 1, 0.5]  # None = non-private (infinity)
EPS_LABEL = {None: "∞ (non-private)", 8: "8", 4: "4", 2: "2", 1: "1", 0.5: "0.5"}


def load_runs():
    runs = [json.loads(l) for l in open(os.path.join(ROOT, "results", "results.jsonl"))]
    return [r for r in runs if r.get("status") == "ok" and r.get("rounds")]


def pick(runs, dataset, eps, compression, lr=None):
    cands = [r for r in runs
             if r["config"].get("dataset") == dataset
             and r["config"].get("target_epsilon") == eps
             and bool(r["config"].get("compression")) == compression
             and r["config"].get("mode", "train") == "train"]
    if lr is not None and dataset == "cifar10" and eps is not None:
        cands = [r for r in cands if abs(r["config"].get("lr", 0.1) - lr) < 1e-9]
    return cands[-1] if cands else None


def final_acc(run):
    return run["rounds"][-1]["accuracy"] if run else None


def avg_uplink_mb(run):
    if not run:
        return None
    vals = [r["uplink_mb"] for r in run["rounds"]]
    return round(sum(vals) / len(vals), 3)


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} ok runs")

    # ---- RQ1/RQ2 table
    rows = ["dataset,epsilon,lr,final_accuracy,wall_clock_s"]
    for ds in ("mnist", "cifar10"):
        for eps in EPS_ORDER:
            r = pick(runs, ds, eps, False)
            if r:
                lr = r["config"].get("lr", 0.1)
                rows.append(f"{ds},{EPS_LABEL[eps]},{lr},{final_acc(r):.4f},{r['wall_clock_s']:.0f}")
            else:
                rows.append(f"{ds},{EPS_LABEL[eps]},,,")
    # CIFAR tuned pass (lr=0.02) reported alongside the collapsed baseline-lr pass
    for eps in EPS_ORDER:
        if eps is None:
            continue
        r = pick(runs, "cifar10", eps, False, lr=0.02)
        if r:
            rows.append(f"cifar10_tuned,{EPS_LABEL[eps]},0.02,{final_acc(r):.4f},{r['wall_clock_s']:.0f}")
    open(os.path.join(ROOT, "results", "summary_table.csv"), "w").write("\n".join(rows))
    print("\n".join(rows))

    # ---- RQ3 table
    rows3 = ["dataset,condition,avg_uplink_mb,final_accuracy"]
    for ds in ("mnist", "cifar10"):
        for eps, tag in ((None, "uncompressed_baseline"), (2, "compressed_eps2"), (8, "compressed_eps8")):
            if eps is None:
                r = pick(runs, ds, None, False)
                cond = "uncompressed"
            else:
                r = pick(runs, ds, eps, True)
                cond = f"compressed (k=10%) eps={eps}"
            if r:
                rows3.append(f"{ds},{cond},{avg_uplink_mb(r)},{final_acc(r):.4f}")
    open(os.path.join(ROOT, "results", "compression_table.csv"), "w").write("\n".join(rows3))
    print()
    print("\n".join(rows3))

    # ---- RQ4 table
    inv = [r for r in runs if r.get("mode") == "inversion"]
    rows4 = ["method,condition,l2,ssim"]
    if inv:
        for row in inv[-1].get("attack_rows", []):
            rows4.append(f"{row['method']},{row['condition']},{row['l2']},{row['ssim']}")
    open(os.path.join(ROOT, "results", "inversion_table.csv"), "w").write("\n".join(rows4))
    print()
    print("\n".join(rows4))

    # ---- Figures 5.1 / 5.2: accuracy vs epsilon curves
    for ds, fname, title in (("mnist", "Figure_5.1_Accuracy_versus_privacy_budget_on_MNIST.png",
                              "MNIST: accuracy versus privacy budget"),
                             ("cifar10", "Figure_5.2_Accuracy_versus_privacy_budget_on_CIFAR-10.png",
                              "CIFAR-10: accuracy versus privacy budget")):
        eps_vals, accs = [], []
        for eps in [0.5, 1, 2, 4, 8]:
            r = pick(runs, ds, eps, False, lr=0.02 if ds == "cifar10" else None)
            if r:
                eps_vals.append(eps)
                accs.append(final_acc(r))
        base = pick(runs, ds, None, False)
        if not eps_vals:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.plot(eps_vals, accs, "o-", color="black", linewidth=1.4, markersize=5, label="FedAvg + central DP")
        if ds == "cifar10":
            col_eps, col_accs = [], []
            for eps in [0.5, 1, 2, 4, 8]:
                r = pick(runs, ds, eps, False, lr=0.1)
                if r:
                    col_eps.append(eps)
                    col_accs.append(final_acc(r))
            if col_eps:
                ax.plot(col_eps, col_accs, "s", color="0.55", markersize=5,
                        label="baseline lr = 0.1 (diverged)")
        if base:
            ax.axhline(final_acc(base), color="black", linestyle="--", linewidth=1.0,
                       label="non-private baseline (ε = ∞)")
        ax.set_xlabel("privacy budget ε", fontsize=10, family="serif")
        ax.set_ylabel("global test accuracy", fontsize=10, family="serif")
        ax.set_title(title, fontsize=11, family="serif")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved", fname)

    # ---- Figure 5.3: communication with/without compression
    labels, up_uncomp, up_comp, ds_list = [], [], [], []
    for ds in ("mnist", "cifar10"):
        u, c = pick(runs, ds, None, False), pick(runs, ds, None, True)
        if u and c:
            ds_list.append(ds.upper())
            up_uncomp.append(avg_uplink_mb(u))
            up_comp.append(avg_uplink_mb(c))
    if ds_list:
        x = range(len(ds_list))
        width = 0.35
        fig, ax = plt.subplots(figsize=(6.4, 4))
        ax.bar([i - width / 2 for i in x], up_uncomp, width, label="uncompressed uplink",
               fill=False, edgecolor="black", linewidth=1.0)
        ax.bar([i + width / 2 for i in x], up_comp, width, label="compressed uplink (k = 10%)",
               fill=False, edgecolor="black", linewidth=1.0, hatch="//")
        ax.set_xticks(list(x))
        ax.set_xticklabels(ds_list, fontsize=10, family="serif")
        ax.set_ylabel("MB per round (4 clients)", fontsize=10, family="serif")
        ax.set_title("Per-round communication with and without compression", fontsize=11, family="serif")
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(IMG, "Figure_5.3_Per-round_communication_with_and_without_compression.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("saved Figure_5.3")


if __name__ == "__main__":
    main()
