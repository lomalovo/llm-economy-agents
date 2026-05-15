"""Multi-criteria θ scoring: rank by J + (predicted) IRF quality.

For each candidate θ (including pert_* and original lhs_*/bo_*), score:
  - J under live FRED targets (calibration error)
  - Predicted IRF quality from sim_moments — does it look "DSGE-like"?
    - Phillips corr sign correct (negative)
    - Beveridge corr sign correct (negative)
    - Okun corr sign correct (negative)
    - Output AR(1) high (>0.8)
    - Inflation std > 0.003 (responsive prices)

Final score: lower J + higher IRF-quality. Both normalized to [0,1].

Usage:
    python scripts/score_theta_irf.py --top-n 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

# Load .env
with open('.env') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            k, v = line.strip().split('=', 1)
            os.environ[k] = v

from src.analysis.fred_targets import get_target_moments
from src.analysis.msm import J


def irf_quality_score(sim_m: dict) -> tuple[float, dict]:
    """Score how 'DSGE-like' the moments look. Higher = better.

    Components (each 0..1, summed):
      phillips_sign: 1 if phillips_corr < 0, else 0
      beveridge_sign: 1 if beveridge_corr < 0, else 0
      okun_sign: 1 if okun_corr < 0, else 0
      output_persistence: min(1, output_ar1 / 0.9)
      inflation_responsiveness: min(1, inflation_std / 0.005)
      unrate_mean_realistic: 1 - min(1, abs(unrate_mean - 0.05) / 0.2)
    """
    parts = {}
    parts["phillips_sign"] = 1.0 if sim_m.get("phillips_corr", 0) < 0 else 0.0
    parts["beveridge_sign"] = 1.0 if sim_m.get("beveridge_corr", 0) < 0 else 0.0
    parts["okun_sign"] = 1.0 if sim_m.get("okun_corr", 0) < 0 else 0.0
    parts["output_persistence"] = min(1.0, max(0.0, sim_m.get("output_ar1", 0) / 0.9))
    parts["inflation_responsiveness"] = min(1.0, sim_m.get("inflation_std", 0) / 0.005)
    parts["unrate_mean_realistic"] = 1.0 - min(1.0, abs(sim_m.get("unrate_mean", 0) - 0.05) / 0.2)
    total = sum(parts.values())
    return total, parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--input", default="data/msm/lhs_results.jsonl")
    p.add_argument("--alpha", type=float, default=0.5, help="weight on J (0..1); rest on IRF quality")
    args = p.parse_args()

    target = get_target_moments()
    with open(args.input) as f:
        records = [json.loads(l) for l in f if l.strip()]

    # Compute J + IRF quality for each
    scored = []
    for r in records:
        j_val, _ = J(r["sim_moments"], target)
        iq_total, iq_parts = irf_quality_score(r["sim_moments"])
        scored.append({**r, "J": j_val, "IRF_quality": iq_total, "IRF_parts": iq_parts})

    # Normalize J (lower = better → 1 - normalized)
    Js = np.array([s["J"] for s in scored])
    J_norm = 1.0 - (Js - Js.min()) / (Js.max() - Js.min() + 1e-9)  # 1 = best
    IQ = np.array([s["IRF_quality"] for s in scored])
    IQ_norm = (IQ - IQ.min()) / (IQ.max() - IQ.min() + 1e-9)  # 1 = best

    for i, s in enumerate(scored):
        s["combined_score"] = args.alpha * J_norm[i] + (1 - args.alpha) * IQ_norm[i]
        s["J_norm"] = float(J_norm[i])
        s["IQ_norm"] = float(IQ_norm[i])

    scored.sort(key=lambda r: r["combined_score"], reverse=True)

    print(f"=== Top-{args.top_n} θ by combined score (α={args.alpha}) ===")
    print(f"{'rank':<5}{'label':<24}{'J':<10}{'IRFq':<7}{'comb':<7}{'A':<6}{'price_adj':<11}{'okun':<8}{'phillips':<10}{'σπ':<8}")
    for i, s in enumerate(scored[:args.top_n]):
        t = s["theta"]; m = s["sim_moments"]
        print(f"{i+1:<5}{s['label']:<24}{s['J']:<10.2f}{s['IRF_quality']:<7.2f}{s['combined_score']:<7.3f}"
              f"{t['match_eff']:<6.2f}{t['price_adj']:<11.2f}{m.get('okun_corr',0):<+8.3f}"
              f"{m.get('phillips_corr',0):<+10.3f}{m.get('inflation_std',0):<8.4f}")

    # Save final ranking
    out = Path("data/msm/theta_ranking.json")
    with open(out, "w") as f:
        json.dump([{"label": s["label"], "J": s["J"], "IRF_quality": s["IRF_quality"],
                    "combined": s["combined_score"], "theta": s["theta"],
                    "IRF_parts": s["IRF_parts"]} for s in scored[:args.top_n]], f, indent=2, default=float)
    print(f"\nSaved ranking to {out}")


if __name__ == "__main__":
    main()
