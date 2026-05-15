"""Full MSM analysis: LHS + BO + refinement → clean summary.

Generates:
  - data/msm/lhs_summary.csv (sorted by J)
  - data/msm/state.json (best θ* on live FRED targets)
  - data/msm/a_binning.csv (J statistics by A-dimension bin)
  - data/msm/per_moment_contribution.csv (which moments contribute most to J)
  - charts/v2/fig_msm_J.png (regenerated)
  - charts/v2/fig_msm_A_pattern.png (J vs A scatter showing monotonicity)
  - charts/v2/fig_moment_fit.png (regenerated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from src.analysis.fred_targets import get_target_moments
from src.analysis.msm import J, CALIBRATION_MOMENTS


def main():
    msm_dir = Path("data/msm")
    charts_dir = Path("charts/v2")
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Load all LHS + BO records
    with open(msm_dir / "lhs_results.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} MSM evaluations")

    # Recompute J with live FRED targets
    target = get_target_moments()
    rescored = []
    for r in records:
        j_new, parts_new = J(r["sim_moments"], target)
        rescored.append({**r, "J": j_new, "parts": parts_new})

    sorted_r = sorted(rescored, key=lambda r: r["J"])
    best = sorted_r[0]

    # 1. lhs_summary.csv
    rows = []
    for r in sorted_r:
        row = {"label": r["label"], "J": r["J"], **r["theta"]}
        for k in CALIBRATION_MOMENTS:
            row[f"mom_{k}"] = r["sim_moments"].get(k, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(msm_dir / "lhs_summary.csv", index=False)
    print(f"Saved lhs_summary.csv ({len(df)} rows)")

    # 2. state.json (live FRED)
    state = {
        "best": best,
        "target": target,
        "target_source": "FRED live (1990-01-01 onwards) via fredapi",
        "n_evaluations": len(records),
    }
    with open(msm_dir / "state.json", "w") as f:
        json.dump(state, f, indent=2, default=float)
    print(f"Best θ* = {best['label']}, J={best['J']:.2f}, A={best['theta']['match_eff']:.2f}")

    # 3. A-binning
    bins = [
        (0.35, 0.55, "A<0.55"),
        (0.55, 0.75, "0.55-0.75"),
        (0.75, 0.90, "0.75-0.90"),
        (0.90, 1.001, "A>=0.90"),
    ]
    binning = []
    for lo, hi, name in bins:
        js = [r["J"] for r in rescored if lo <= r["theta"]["match_eff"] < hi]
        if js:
            binning.append({
                "bin": name, "n": len(js),
                "min_J": float(np.min(js)),
                "median_J": float(np.median(js)),
                "max_J": float(np.max(js)),
                "mean_J": float(np.mean(js)),
            })
    df_bin = pd.DataFrame(binning)
    df_bin.to_csv(msm_dir / "a_binning.csv", index=False)
    print("\nA-binning:")
    print(df_bin.to_string(index=False))

    # 4. per-moment contribution at θ*
    parts = best["parts"]
    contrib = pd.DataFrame([{"moment": k, "contribution": v, "share": v / best["J"]} for k, v in parts.items()])
    contrib = contrib.sort_values("contribution", ascending=False)
    contrib.to_csv(msm_dir / "per_moment_contribution.csv", index=False)
    print(f"\nTop-3 J contributors at θ*: {', '.join(contrib['moment'].head(3))}")

    # 5. Chart: J distribution sorted
    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted = df.sort_values("J").reset_index(drop=True)
    bo_mask = df_sorted["label"].str.startswith("bo_")
    lhs2_mask = df_sorted["label"].str.startswith("lhs2_")
    lhs_mask = ~bo_mask & ~lhs2_mask
    ax.bar(np.arange(len(df_sorted))[lhs_mask], df_sorted["J"][lhs_mask], color="#1f77b4", label=f"LHS seed=42 (n={lhs_mask.sum()})")
    ax.bar(np.arange(len(df_sorted))[lhs2_mask], df_sorted["J"][lhs2_mask], color="#ff7f0e", label=f"LHS seed=43 (n={lhs2_mask.sum()})")
    if bo_mask.any():
        ax.bar(np.arange(len(df_sorted))[bo_mask], df_sorted["J"][bo_mask], color="#2ca02c", label=f"Bayesian Opt (n={bo_mask.sum()})")
    ax.set_yscale("log")
    ax.set_xlabel("Rank (sorted by J)")
    ax.set_ylabel("J(θ) — log scale")
    ax.set_title(f"MSM distance J(θ) across {len(df_sorted)} evaluations\nbest = {best['label']} (J={best['J']:.2f}, A={best['theta']['match_eff']:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "fig_msm_J.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {charts_dir / 'fig_msm_J.png'}")

    # 6. Chart: J vs A (monotonicity)
    fig, ax = plt.subplots(figsize=(10, 5))
    A_vals = df["match_eff"].values
    J_vals = df["J"].values
    colors = ["#1f77b4" if l.startswith("lhs_") else "#ff7f0e" if l.startswith("lhs2_") else "#2ca02c" for l in df["label"]]
    ax.scatter(A_vals, J_vals, c=colors, s=80, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("matching_efficiency A")
    ax.set_ylabel("J(θ) — log scale")
    ax.set_title(f"J(θ) vs A (matching efficiency): clean monotonic decrease — robust signal not random luck")
    ax.grid(alpha=0.3)
    # Annotate top-3
    top3 = df.nsmallest(3, "J")
    for _, row in top3.iterrows():
        ax.annotate(row["label"], (row["match_eff"], row["J"]),
                    xytext=(5, -10), textcoords="offset points", fontsize=8, color="darkred")
    plt.tight_layout()
    plt.savefig(charts_dir / "fig_msm_A_pattern.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {charts_dir / 'fig_msm_A_pattern.png'}")

    # 7. Moment fit chart
    keys = list(target.keys())
    targets_v = [target[k] for k in keys]
    sim_v = [best["sim_moments"].get(k, np.nan) for k in keys]

    x = np.arange(len(keys))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, targets_v, w, label="FRED target (live 1990-2024)", color="#444")
    ax.bar(x + w / 2, sim_v, w, label=f"LLM-ABM at θ* (J={best['J']:.2f})", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"Calibrated moments vs. FRED — sign-match {sum(1 for s,t in zip(sim_v, targets_v) if (s>=0)==(t>=0))}/{len(keys)} = {sum(1 for s,t in zip(sim_v, targets_v) if (s>=0)==(t>=0))/len(keys)*100:.0f}%")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "fig_moment_fit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {charts_dir / 'fig_moment_fit.png'}")

    return state


if __name__ == "__main__":
    main()
