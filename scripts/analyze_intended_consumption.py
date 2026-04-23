"""Post-process audit JSONL to extract INTENDED consumption per step per condition.

This complements analyze_counterfactual.py which uses realised `last_spent`
(supply-constrained). Intended `consumption_budget` is the pure MPC signal and
often more informative for HANK-style counterfactual tests when the aggregate
goods market is supply-constrained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def extract_intended(audit_path: Path, shock_step: int = 10) -> pd.DataFrame:
    records = []
    with open(audit_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("agent_type") != "household":
                continue
            if "decision" not in r:
                continue
            records.append({
                "step": r["step"],
                "agent_id": r["agent_id"],
                "intended_consumption": r["decision"].get("consumption_budget", 0),
                "intended_labor":       r["decision"].get("labor_supply", 0),
                "intended_savings":     r["decision"].get("savings_amount", 0),
                "actual_spent":         r.get("outcome", {}).get("spent", 0),
                "actual_worked":        r.get("outcome", {}).get("worked", 0),
            })
    return pd.DataFrame(records)


def aggregate_per_step(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("step").agg(
        total_intended=("intended_consumption", "sum"),
        total_actual=("actual_spent", "sum"),
        total_intended_labor=("intended_labor", "sum"),
        total_actual_worked=("actual_worked", "sum"),
        total_intended_savings=("intended_savings", "sum"),
        n_agents=("agent_id", "count"),
    ).reset_index()
    return agg


def irf_vs_baseline(agg: pd.DataFrame, shock_step: int) -> pd.DataFrame:
    """Subtract pre-shock mean of intended/actual from each column."""
    pre = agg[(agg["step"] >= shock_step - 3) & (agg["step"] < shock_step)]
    baseline_intended = pre["total_intended"].mean()
    baseline_actual   = pre["total_actual"].mean()
    baseline_labor    = pre["total_intended_labor"].mean()
    baseline_savings  = pre["total_intended_savings"].mean()

    agg = agg.copy()
    agg["intended_irf"]      = agg["total_intended"]      - baseline_intended
    agg["actual_irf"]        = agg["total_actual"]        - baseline_actual
    agg["labor_irf"]         = agg["total_intended_labor"] - baseline_labor
    agg["savings_irf"]       = agg["total_intended_savings"] - baseline_savings
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-dir", default="data/results",
                   help="directory containing *_reasoning.jsonl")
    p.add_argument("--conditions", nargs="+",
                   default=["cf_all_htm", "cf_all_saver", "cf_mixed", "cf_fifty_fifty"])
    p.add_argument("--shock-step", type=int, default=10)
    p.add_argument("--out-dir", default="data/counterfactual/intended")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cond in args.conditions:
        files = sorted(Path(args.audit_dir).glob(f"{cond}_*_reasoning.jsonl"))
        if not files:
            print(f"{cond}: no files")
            continue
        # Concatenate all runs, then average per step across runs
        dfs = []
        for f in files:
            df_run = extract_intended(f, args.shock_step)
            df_run["run"] = str(f.stem)
            dfs.append(df_run)
        all_df = pd.concat(dfs, ignore_index=True)

        # Aggregate per step per run, then average across runs
        per_run_agg = all_df.groupby(["run", "step"]).agg(
            total_intended=("intended_consumption", "sum"),
            total_actual=("actual_spent", "sum"),
            total_intended_labor=("intended_labor", "sum"),
            total_intended_savings=("intended_savings", "sum"),
        ).reset_index()

        # Mean across runs
        cross_run = per_run_agg.groupby("step").mean(numeric_only=True).reset_index()
        cross_run_irf = irf_vs_baseline(cross_run, args.shock_step)
        cross_run_irf.to_csv(out_dir / f"{cond}_intended.csv", index=False)

        post = cross_run_irf[cross_run_irf["step"] >= args.shock_step]
        summary[cond] = {
            "n_runs": len(files),
            "pre_intended":   float(cross_run_irf[cross_run_irf["step"] < args.shock_step]["total_intended"].mean()),
            "peak_intended_irf": float(post["intended_irf"].abs().max()),
            "mean_intended_irf_post": float(post["intended_irf"].mean()),
            "mean_actual_irf_post":   float(post["actual_irf"].mean()),
            "step10_intended_irf":    float(cross_run_irf[cross_run_irf["step"] == args.shock_step]["intended_irf"].iloc[0])
                                       if not cross_run_irf[cross_run_irf["step"] == args.shock_step].empty else None,
        }
        print(f"{cond}: n_runs={len(files)}  peak_intended_irf={summary[cond]['peak_intended_irf']:.2f}  "
              f"mean_actual_post={summary[cond]['mean_actual_irf_post']:.2f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
