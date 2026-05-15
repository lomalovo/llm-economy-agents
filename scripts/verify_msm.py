"""Verify MSM result is not random:
  1. Pin best θ* and re-run N times → distribution of J
  2. Vary A holding everything else at θ* → monotonicity check

Usage:
    python scripts/verify_msm.py --mode replicate --n-runs 5
    python scripts/verify_msm.py --mode grid-A
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.analysis.fred_targets import get_target_moments
from src.analysis.moments import moments_from_sim_csv
from src.analysis.msm import J, deep_merge, theta_to_config_patch
from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config


async def _run(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


def override_hh(cfg: dict, n: int) -> dict:
    cfg = deepcopy(cfg)
    for g in cfg.get("agents", {}).values():
        if g.get("type") == "household":
            g["count"] = n
            break
    return cfg


async def replicate(args):
    """Re-run best θ* N times, report J distribution."""
    with open(args.state) as f:
        state = json.load(f)
    theta = state["best"]["theta"]
    target = state["target"]

    base = load_config(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "replicate_results.jsonl"

    print(f"Replicating θ* = {theta}")
    print(f"target J was {state['best']['J']:.2f}")

    js = []
    for i in range(args.n_runs):
        cfg = deep_merge(base, theta_to_config_patch(theta))
        cfg = override_hh(cfg, args.households)
        cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"verify_replicate_{i:02d}"}
        try:
            df = await _run(cfg, args.steps)
            sim_m = moments_from_sim_csv(df)
            j_val, _ = J(sim_m, target)
            js.append(j_val)
            rec = {"run": i, "J": j_val, "moments": sim_m}
            with open(log, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"  run {i}: J={j_val:.2f}  unrate={sim_m.get('unrate_mean', 0):.3f}")
        except Exception as e:
            print(f"  run {i}: FAILED — {e}")

    if js:
        arr = np.array(js)
        summary = {
            "n": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        with open(out_dir / "replicate_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nDistribution of J at θ*: mean={summary['mean']:.2f}, median={summary['median']:.2f}, std={summary['std']:.2f}")


async def grid_A(args):
    """Hold θ* fixed except A, vary A across grid, observe J(A)."""
    with open(args.state) as f:
        state = json.load(f)
    theta_base = state["best"]["theta"]
    target = state["target"]

    base = load_config(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "grid_A_results.jsonl"

    A_values = [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    print(f"Grid over A: {A_values}")
    print(f"Other θ fixed: phi_pi={theta_base['phi_pi']:.2f}, "
          f"phi_u={theta_base['phi_u']:.2f}, "
          f"price_adj={theta_base['price_adj']:.2f}, "
          f"sep={theta_base['separation']:.3f}")

    for A in A_values:
        theta = {**theta_base, "match_eff": A}
        cfg = deep_merge(base, theta_to_config_patch(theta))
        cfg = override_hh(cfg, args.households)
        cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"verify_gridA_A{int(A*100):03d}"}
        try:
            df = await _run(cfg, args.steps)
            sim_m = moments_from_sim_csv(df)
            j_val, parts = J(sim_m, target)
            rec = {"A": A, "J": j_val, "moments": sim_m, "parts": parts}
            with open(log, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"  A={A:.2f}: J={j_val:.2f}  unrate={sim_m.get('unrate_mean', 0):.3f}  okun={sim_m.get('okun_corr', 0):+.3f}")
        except Exception as e:
            print(f"  A={A:.2f}: FAILED — {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["replicate", "grid-A"], required=True)
    p.add_argument("--state", default="data/msm/state.json")
    p.add_argument("--base", default="config/experiments/msm_base.yaml")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--households", type=int, default=20)
    p.add_argument("--out", default="data/msm_verify")
    args = p.parse_args()
    if args.mode == "replicate":
        asyncio.run(replicate(args))
    else:
        asyncio.run(grid_A(args))


if __name__ == "__main__":
    main()
