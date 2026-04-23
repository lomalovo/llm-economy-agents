"""Generate paper-ready figures from collected experiment data.

Produces:
    fig_msm_J.png         — J(θ) distribution across LHS points (sorted)
    fig_moment_fit.png    — calibrated moments vs. FRED targets (bar chart)
    fig_scale.png         — stylized fact error vs. N (scale convergence)
    fig_counterfactual.png — output IRF by bio composition
    fig_narrative.png     — reasoning consistency by step / archetype
    fig_irf_demand.png    — calibrated demand shock IRF with CI (reuses bootstrap_irf)
    fig_irf_prod.png      — calibrated productivity shock IRF with CI

Only generates figures for data that exists. Missing data is skipped with a
warning.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHARTS_DIR = Path("charts/v2")
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def fig_msm_J(msm_dir: Path = Path("data/msm")):
    csv = msm_dir / "lhs_summary.csv"
    if not csv.exists():
        print(f"[skip] {csv} not found")
        return
    df = pd.read_csv(csv).sort_values("J").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["J"].values, "o-", color="#1f77b4")
    ax.set_xlabel("Rank (sorted)")
    ax.set_ylabel("J(θ)")
    ax.set_title(f"MSM distance across LHS grid (best J={df['J'].min():.2f})")
    ax.grid(alpha=0.3)
    out = CHARTS_DIR / "fig_msm_J.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_moment_fit(msm_dir: Path = Path("data/msm")):
    state = msm_dir / "state.json"
    if not state.exists():
        print(f"[skip] {state} not found")
        return
    with open(state) as f:
        st = json.load(f)
    best = st["best"]
    target = st["target"]
    sim = best["sim_moments"]
    keys = [k for k in target if not (isinstance(target[k], float) and np.isnan(target[k]))]
    targets_v = [target[k] for k in keys]
    sim_v = [sim.get(k, float("nan")) for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, targets_v, w, label="FRED target", color="#444")
    ax.bar(x + w / 2, sim_v, w, label="Calibrated LLM-ABM", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Calibrated moments vs. FRED target (θ*={best['theta']})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = CHARTS_DIR / "fig_moment_fit.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_scale(scale_csv: Path = Path("data/scale_ablation/scale_summary.csv")):
    if not scale_csv.exists():
        print(f"[skip] {scale_csv} not found")
        return
    df = pd.read_csv(scale_csv)
    metrics = [
        ("okun_corr", -0.72),
        ("phillips_corr", -0.145),
        ("beveridge_corr", -0.835),
        ("unrate_ar1", 0.95),
        ("output_ar1", 0.996),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    for key, target in metrics:
        if key not in df.columns:
            continue
        err = np.abs(df[key] - target)
        ax.plot(df["n"], err, "o-", label=f"{key} (target={target:+.2f})")
    ax.set_xscale("log")
    ax.set_xticks(df["n"])
    ax.set_xticklabels(df["n"].astype(int))
    ax.set_xlabel("N (households)")
    ax.set_ylabel("|sim − target|")
    ax.set_title("Scale convergence (dummy backend)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    out = CHARTS_DIR / "fig_scale.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_counterfactual(cf_dir: Path = Path("data/counterfactual/analysis")):
    summary = cf_dir / "summary.json"
    if not summary.exists():
        print(f"[skip] {summary} not found")
        return
    with open(summary) as f:
        s = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    conditions = [c for c in ["cf_all_saver", "cf_mixed", "cf_fifty_fifty", "cf_all_htm"] if c in s]
    labels = [c.replace("cf_", "") for c in conditions]

    for ax, var in zip(axes, ["total_sales", "total_consumption"]):
        for cond, label in zip(conditions, labels):
            irf_csv = cf_dir / f"{cond}_irf.csv"
            if not irf_csv.exists():
                continue
            irf = pd.read_csv(irf_csv, index_col=0)
            ax.plot(irf.index, irf[var], label=label, linewidth=2)
        ax.axvline(10, color="red", linestyle=":", linewidth=1, label="shock (t=10)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"Counterfactual IRF: {var}")
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    out = CHARTS_DIR / "fig_counterfactual.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def fig_narrative(audit_dir: Path = Path("data/audit")):
    f = audit_dir / "summary.json"
    if not f.exists():
        print(f"[skip] {f} not found")
        return
    with open(f) as fh:
        data = json.load(fh)
    if not data:
        return

    counts = data.get("counts", {})
    by_group = data.get("by_group", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: aggregate
    label_order = ["consistent", "inconsistent", "unclear", "error"]
    labels = [l for l in label_order if l in counts]
    vals = [counts[l] for l in labels]
    colors = {"consistent": "#2ca02c", "inconsistent": "#d62728", "unclear": "#ff7f0e", "error": "#999"}
    axes[0].bar(labels, vals, color=[colors[l] for l in labels])
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Aggregate (n={data.get('total', sum(vals))})")
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    # Right: consistency rate by condition
    if by_group:
        group_labels = sorted(by_group.keys())
        consistent_rates = []
        for g in group_labels:
            c = by_group[g]
            total = sum(c.values())
            consistent_rates.append(c.get("consistent", 0) / total * 100 if total else 0)
        axes[1].bar(group_labels, consistent_rates, color="#2ca02c")
        axes[1].set_ylabel("% consistent")
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Consistency rate by bio composition")
        for lbl in axes[1].get_xticklabels():
            lbl.set_rotation(20)
            lbl.set_ha("right")
        for i, v in enumerate(consistent_rates):
            axes[1].text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    out = CHARTS_DIR / "fig_narrative.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def main():
    fig_msm_J()
    fig_moment_fit()
    fig_scale()
    fig_counterfactual()
    fig_narrative()
    print(f"\nAll figures saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
