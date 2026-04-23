"""MSM calibration driver.

Stages:
  1. LHS grid over θ, one run per point, compute moments, compute J.
  2. Pick argmin θ*, run N_refine times to reduce run-to-run noise.
  3. Report best θ*, final moments vs target, J decomposition.

Checkpoints after every run so partial results survive interruptions.

Usage:
    python run_msm.py --base config/experiments/baseline.yaml --lhs 15 --steps 60 --households 30
    python run_msm.py --refine results/msm/state.json --refine-runs 3 --steps 80 --households 50
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.fred_targets import get_target_moments
from src.analysis.moments import moments_from_multiple_runs, moments_from_sim_csv
from src.analysis.msm import (
    CALIBRATION_MOMENTS,
    J,
    deep_merge,
    find_best,
    latin_hypercube,
    theta_to_config_patch,
)
from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config


# Default search space (bounds) — edit to change what MSM optimizes over.
DEFAULT_BOUNDS = {
    "phi_pi":     (0.5, 2.5),
    "phi_u":      (0.0, 1.2),
    "match_eff":  (0.35, 1.0),
    "price_adj":  (0.10, 1.0),
    "separation": (0.0, 0.15),
}


async def _run_single(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


def override_household_count(cfg: dict, n: int) -> dict:
    cfg = deepcopy(cfg)
    for group_name, group in cfg.get("agents", {}).items():
        if group.get("type") == "household":
            group["count"] = n
            break
    return cfg


async def evaluate_theta(
    theta: dict,
    base_cfg: dict,
    steps: int,
    households: int | None,
    target_moments: dict,
    label: str = "",
) -> dict:
    """Run one simulation with the given θ, compute J, return a record."""
    patch = theta_to_config_patch(theta)
    cfg = deep_merge(base_cfg, patch)
    if households is not None:
        cfg = override_household_count(cfg, households)
    # Unique experiment name so logs do not collide
    cfg["experiment"] = deepcopy(cfg.get("experiment", {}))
    cfg["experiment"]["name"] = f"msm_{label or int(time.time())}"

    t0 = time.time()
    df = await _run_single(cfg, steps)
    elapsed = time.time() - t0

    sim_moments = moments_from_sim_csv(df)
    j_val, parts = J(sim_moments, target_moments)
    return {
        "theta":        theta,
        "J":            j_val,
        "parts":        parts,
        "sim_moments":  sim_moments,
        "target":       target_moments,
        "elapsed_s":    round(elapsed, 1),
        "n_steps":      steps,
        "n_households": households,
        "label":        label,
    }


def _append_checkpoint(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=float) + "\n")


def _load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_lhs_grid(
    base_cfg: dict,
    n_points: int,
    steps: int,
    households: int,
    bounds: dict,
    target: dict,
    out_dir: Path,
    resume: bool = True,
) -> list[dict]:
    checkpoint = out_dir / "lhs_results.jsonl"
    thetas = latin_hypercube(n_points, bounds, seed=42)

    existing = _load_checkpoint(checkpoint) if resume else []
    done_labels = {r["label"] for r in existing}

    results = list(existing)
    for i, theta in enumerate(thetas):
        label = f"lhs_{i:03d}"
        if label in done_labels:
            print(f"  [{label}] already computed, skipping")
            continue
        print(f"\n=== LHS {i + 1}/{n_points}  θ = {theta} ===")
        try:
            rec = await evaluate_theta(theta, base_cfg, steps, households, target, label)
            results.append(rec)
            _append_checkpoint(checkpoint, rec)
            print(f"  J = {rec['J']:.3f}  moments = {rec['sim_moments']}")
        except Exception as e:
            print(f"  [FAIL] {e}")
            _append_checkpoint(out_dir / "failures.jsonl",
                               {"label": label, "theta": theta, "error": str(e)})
    return results


async def refine_best(
    best: dict,
    base_cfg: dict,
    steps: int,
    households: int,
    target: dict,
    n_runs: int,
    out_dir: Path,
) -> list[dict]:
    theta = best["theta"]
    results = []
    for i in range(n_runs):
        label = f"refine_{i:02d}"
        print(f"\n=== REFINE {i + 1}/{n_runs}  θ* = {theta} ===")
        rec = await evaluate_theta(theta, base_cfg, steps, households, target, label)
        results.append(rec)
        _append_checkpoint(out_dir / "refine_results.jsonl", rec)
        print(f"  J = {rec['J']:.3f}")
    return results


def summarize(results: list[dict], out_dir: Path, tag: str = "summary"):
    if not results:
        print("No results to summarize")
        return
    rows = []
    for r in results:
        row = {"J": r["J"], "label": r["label"], **r["theta"]}
        for k in CALIBRATION_MOMENTS:
            row[f"mom_{k}"] = r["sim_moments"].get(k, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("J").reset_index(drop=True)
    out_csv = out_dir / f"{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(df)} results to {out_csv}")
    print("\nTop 5 by J:")
    print(df.head().to_string(index=False))
    return df


async def main_async(args: argparse.Namespace):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base)
    target = get_target_moments()
    print("Target moments (FRED):")
    for k, v in target.items():
        print(f"  {k}: {v:+.4f}")

    # Save target moments to the output dir for reproducibility
    with open(out_dir / "target_moments.json", "w") as f:
        json.dump(target, f, indent=2)

    if args.refine:
        # Load state.json to get best θ
        best_path = Path(args.refine)
        with open(best_path) as f:
            state = json.load(f)
        best = state["best"]
        results = await refine_best(
            best, base_cfg, args.steps, args.households, target,
            args.refine_runs, out_dir,
        )
        summarize(results, out_dir, tag="refine_summary")
        return

    lhs_results = await run_lhs_grid(
        base_cfg, args.lhs, args.steps, args.households,
        DEFAULT_BOUNDS, target, out_dir, resume=True,
    )
    df_lhs = summarize(lhs_results, out_dir, tag="lhs_summary")

    if lhs_results:
        best = find_best(lhs_results)
        state = {
            "best":   best,
            "target": target,
            "bounds": DEFAULT_BOUNDS,
        }
        with open(out_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2, default=float)
        print(f"\nBest θ* = {best['theta']}")
        print(f"Best J   = {best['J']:.3f}")
        print(f"State saved to {out_dir / 'state.json'}")


def main():
    p = argparse.ArgumentParser(description="MSM calibration")
    p.add_argument("--base", default="config/experiments/baseline.yaml")
    p.add_argument("--lhs", type=int, default=15)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--households", type=int, default=30)
    p.add_argument("--out", default="data/msm")
    p.add_argument("--refine", default=None, help="Path to state.json to refine its best θ")
    p.add_argument("--refine-runs", type=int, default=3)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
