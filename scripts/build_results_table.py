"""Consolidate all experimental artefacts into a single paper-ready results table.

Sources:
  data/scale_ablation/scale_summary.csv
  data/counterfactual/analysis/summary.json
  data/msm/state.json
  data/msm/refine_summary.csv (if refined)
  data/audit/summary.json
  data/hero/*.csv (via bootstrap)

Outputs:
  data/final_results.json          — machine-readable
  data/final_results.md             — human-readable summary for paper
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


def load_if_exists(path, reader=json.load):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return reader(f)


def main():
    out = {}

    # 1. MSM calibration
    state = load_if_exists("data/msm/state.json")
    if state:
        best = state["best"]
        target = state["target"]
        out["msm"] = {
            "best_theta": best["theta"],
            "best_J": best["J"],
            "sim_moments": best["sim_moments"],
            "target_moments": target,
            "per_moment_contribution": best["parts"],
        }

        # Compute RMSE and sign agreement
        keys = list(target.keys())
        deviations = []
        sign_matches = 0
        for k in keys:
            if k in best["sim_moments"]:
                s = best["sim_moments"][k]
                t = target[k]
                if not (isinstance(s, float) and np.isnan(s)):
                    deviations.append((s - t) ** 2)
                    if (s >= 0) == (t >= 0):
                        sign_matches += 1
        out["msm"]["rmse"] = float(np.sqrt(np.mean(deviations))) if deviations else None
        out["msm"]["sign_match_rate"] = sign_matches / len(keys) if keys else None

    # 2. Counterfactual
    cf = load_if_exists("data/counterfactual/analysis/summary.json")
    if cf:
        out["counterfactual"] = cf
        # HANK test: monotonicity and HtM vs saver delta
        if all(c in cf for c in ["cf_all_htm", "cf_all_saver"]):
            htm = cf["cf_all_htm"]
            saver = cf["cf_all_saver"]
            out["counterfactual"]["hank_test"] = {
                "delta_output_mean":  htm["mean_output_irf"] - saver["mean_output_irf"],
                "delta_consumption_mean": htm["mean_consumption_irf"] - saver["mean_consumption_irf"],
            }

    # 3. Narrative audit — reconstruct totals from by_group (authoritative source of all records)
    audit = load_if_exists("data/audit/summary.json")
    if audit:
        # by_group sums are the actual ground truth (merged input × judgments)
        by_group = audit.get("by_group", {})
        if by_group:
            total_counts = {"consistent": 0, "inconsistent": 0, "unclear": 0, "error": 0, "missing": 0}
            for group_counts in by_group.values():
                for k, v in group_counts.items():
                    total_counts[k] = total_counts.get(k, 0) + v
            total_n = sum(total_counts.values())
            rates = {k: (v / total_n if total_n else 0.0) for k, v in total_counts.items()}
            audit["counts"] = total_counts
            audit["rates"] = rates
            audit["total"] = total_n
        out["narrative"] = audit

    # 4. Scale ablation
    scale_csv = Path("data/scale_ablation/scale_summary.csv")
    if scale_csv.exists():
        df = pd.read_csv(scale_csv)
        out["scale"] = {
            "n_values": df["n"].tolist(),
            "okun_corr_by_N": dict(zip(df["n"].tolist(), df["okun_corr"].tolist())),
            "beveridge_corr_by_N": dict(zip(df["n"].tolist(), df["beveridge_corr"].tolist())),
            "phillips_corr_by_N": dict(zip(df["n"].tolist(), df["phillips_corr"].tolist())),
            "inflation_std_by_N": dict(zip(df["n"].tolist(), df["inflation_std"].tolist())),
        }

    # Save machine-readable
    Path("data").mkdir(exist_ok=True)
    with open("data/final_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Human-readable markdown
    md = ["# LLM-ABM Paper v2 — Final Results\n"]

    if "msm" in out:
        md.append("## MSM calibration\n")
        md.append(f"- Best θ*: `{out['msm']['best_theta']}`")
        md.append(f"- Best J: **{out['msm']['best_J']:.3f}**")
        md.append(f"- RMSE vs FRED targets: **{out['msm']['rmse']:.4f}**")
        md.append(f"- Sign-match rate: **{out['msm']['sign_match_rate']:.1%}**\n")
        md.append("| Moment | FRED target | Sim (θ*) | Contribution |")
        md.append("|---|---|---|---|")
        for k in out['msm']['target_moments']:
            t = out['msm']['target_moments'][k]
            s = out['msm']['sim_moments'].get(k, float("nan"))
            c = out['msm']['per_moment_contribution'].get(k, 0)
            md.append(f"| {k} | {t:+.4f} | {s:+.4f} | {c:.2f} |")

    if "counterfactual" in out:
        md.append("\n## Counterfactual bio composition\n")
        md.append("| Condition | n_runs | Mean ΔConsumption | Mean ΔOutput |")
        md.append("|---|---|---|---|")
        for cond in ["cf_all_saver", "cf_mixed", "cf_fifty_fifty", "cf_all_htm"]:
            if cond in out["counterfactual"]:
                s = out["counterfactual"][cond]
                md.append(f"| {cond} | {s['n_runs']} | {s['mean_consumption_irf']:+.3f} | {s['mean_output_irf']:+.3f} |")
        if "hank_test" in out["counterfactual"]:
            h = out["counterfactual"]["hank_test"]
            md.append(f"\nHANK test: ΔYmean(HtM) − ΔYmean(saver) = **{h['delta_output_mean']:+.3f}**, "
                      f"ΔCmean(HtM) − ΔCmean(saver) = **{h['delta_consumption_mean']:+.3f}**")

    if "narrative" in out:
        md.append("\n## Narrative causal audit\n")
        n = out["narrative"]
        md.append(f"- Total records audited: {n.get('total', 0)}")
        md.append(f"- Consistency rate: **{n.get('rates', {}).get('consistent', 0):.1%}**")
        md.append(f"- Inconsistent: {n.get('rates', {}).get('inconsistent', 0):.1%}")
        md.append(f"- Unclear: {n.get('rates', {}).get('unclear', 0):.1%}")

    if "scale" in out:
        md.append("\n## Scale convergence\n")
        md.append("| N | Okun | Beveridge | Phillips | σ(π) |")
        md.append("|---|---|---|---|---|")
        for n in out['scale']['n_values']:
            md.append(
                f"| {n} | {out['scale']['okun_corr_by_N'][n]:+.3f} | "
                f"{out['scale']['beveridge_corr_by_N'][n]:+.3f} | "
                f"{out['scale']['phillips_corr_by_N'][n]:+.3f} | "
                f"{out['scale']['inflation_std_by_N'][n]:.4f} |"
            )
        md.append("\nTargets (FRED): Okun=−0.72, Beveridge=−0.835, Phillips=−0.145, σ(π)≈0.007")

    with open("data/final_results.md", "w") as f:
        f.write("\n".join(md))
    print("Wrote data/final_results.json and data/final_results.md")


if __name__ == "__main__":
    main()
