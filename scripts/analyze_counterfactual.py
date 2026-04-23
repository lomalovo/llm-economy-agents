"""Analyze counterfactual bio experiments.

Loads all cf_* CSVs from data/counterfactual/, aligns by step, computes for each
condition:
  1. Average trajectory of key macro variables (output, consumption, inflation, unemployment)
  2. Impulse response on shock at step 10 (relative to each condition's pre-shock baseline)
  3. Peak IRF at steps 11-20 across runs
  4. Cross-condition hypothesis test: does HtM population amplify the shock more than saver population?

Kaplan–Violante prediction: 100%-HtM > 50/50 > mixed > 100%-saver in |output IRF|.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ["cf_all_htm", "cf_all_saver", "cf_mixed", "cf_fifty_fifty"]
TRACKED_VARS = ["total_sales", "total_consumption", "inflation_rate",
                "unemployment_rate", "vacancy_rate", "real_wage", "interest_rate"]


def load_runs(cond: str, data_dir: Path) -> list[pd.DataFrame]:
    files = sorted(data_dir.glob(f"{cond}_run_*.csv"))
    return [pd.read_csv(f) for f in files]


def compute_irf(runs: list[pd.DataFrame], shock_step: int = 10) -> pd.DataFrame:
    """Return DataFrame indexed by step, one column per variable = mean across runs,
    with *pre-shock mean subtracted* (so step=shock_step onwards is IRF deviation)."""
    if not runs:
        return pd.DataFrame()
    aligned = {}
    for v in TRACKED_VARS:
        arr = np.stack([r[v].values for r in runs if v in r.columns], axis=0)
        mean_ts = arr.mean(axis=0)
        # Subtract mean of last 3 pre-shock steps as baseline
        baseline_window = (shock_step - 3, shock_step)
        baseline = mean_ts[baseline_window[0] : baseline_window[1]].mean()
        aligned[v] = mean_ts - baseline
    steps = runs[0]["step"].values
    df = pd.DataFrame(aligned, index=steps)
    return df


def summarize_counterfactual(data_dir: Path, shock_step: int = 10) -> dict:
    summaries = {}
    for cond in CONDITIONS:
        runs = load_runs(cond, data_dir)
        if not runs:
            print(f"  {cond}: no runs found")
            continue
        irf = compute_irf(runs, shock_step)
        # Peak |output| IRF in post-shock window
        post_shock_window = (shock_step, shock_step + 10)
        peak_row = irf.loc[post_shock_window[0] : post_shock_window[1]]
        summaries[cond] = {
            "n_runs": len(runs),
            "irf_df": irf,
            "peak_output_irf":   float(peak_row["total_sales"].abs().max()),
            "peak_consumption_irf": float(peak_row["total_consumption"].abs().max()),
            "mean_output_irf":   float(peak_row["total_sales"].mean()),
            "mean_consumption_irf": float(peak_row["total_consumption"].mean()),
            "final_output_irf":  float(irf["total_sales"].iloc[-1]),
            "kleinberg_pre":  float(irf.loc[shock_step - 3 : shock_step - 1]["total_sales"].mean()),
        }
    return summaries


def print_report(summaries: dict):
    print("\n======================= COUNTERFACTUAL BIO ANALYSIS =======================")
    print(f"{'condition':<18} {'n_runs':>6} {'mean ΔC':>10} {'peak |ΔC|':>12} {'mean ΔY':>10} {'peak |ΔY|':>12}")
    print("-" * 80)
    for cond, s in summaries.items():
        print(
            f"{cond:<18} {s['n_runs']:>6} "
            f"{s['mean_consumption_irf']:>+10.3f} {s['peak_consumption_irf']:>12.3f} "
            f"{s['mean_output_irf']:>+10.3f} {s['peak_output_irf']:>12.3f}"
        )
    print("-" * 80)

    # Compare all_htm vs all_saver
    if "cf_all_htm" in summaries and "cf_all_saver" in summaries:
        h = summaries["cf_all_htm"]
        s = summaries["cf_all_saver"]
        dc = h["mean_consumption_irf"] - s["mean_consumption_irf"]
        dy = h["mean_output_irf"] - s["mean_output_irf"]
        print(f"\nHtM vs saver deltas (positive → HtM amplifies shock):")
        print(f"  ΔCmean(HtM) − ΔCmean(saver) = {dc:+.3f}")
        print(f"  ΔYmean(HtM) − ΔYmean(saver) = {dy:+.3f}")

    # Monotonicity check across conditions if all four present
    required = [c for c in CONDITIONS if c in summaries]
    if len(required) == 4:
        # Order by HtM-fraction: all_saver(0) -> mixed(0.3) -> 50/50(0.5) -> all_htm(1.0)
        ordered = ["cf_all_saver", "cf_mixed", "cf_fifty_fifty", "cf_all_htm"]
        mean_c_irfs = [summaries[c]["mean_consumption_irf"] for c in ordered]
        monotonic = all(mean_c_irfs[i] <= mean_c_irfs[i + 1] for i in range(3))
        print(f"\nMonotonicity across HtM-fraction (saver→mixed→50/50→HtM):")
        print(f"  mean ΔC: {mean_c_irfs}  → {'MONOTONIC' if monotonic else 'NOT monotonic'}")


def save_artifacts(summaries: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save IRFs as CSV
    for cond, s in summaries.items():
        s["irf_df"].to_csv(out_dir / f"{cond}_irf.csv")
    # Save scalar summaries
    json_summary = {
        cond: {k: v for k, v in s.items() if k != "irf_df"}
        for cond, s in summaries.items()
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(json_summary, f, indent=2, default=float)
    print(f"\nArtifacts saved to {out_dir}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/counterfactual")
    p.add_argument("--out", default="data/counterfactual/analysis")
    p.add_argument("--shock-step", type=int, default=10)
    args = p.parse_args()

    summaries = summarize_counterfactual(Path(args.data), args.shock_step)
    print_report(summaries)
    save_artifacts(summaries, Path(args.out))


if __name__ == "__main__":
    main()
