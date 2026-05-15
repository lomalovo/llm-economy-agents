"""Smart θ search: perturb around known-good clusters from prior MSM.

Approach:
  1. Identify K cluster centroids in existing top-N MSM evaluations
  2. For each centroid, sample M perturbed θ via Gaussian noise scaled by cluster spread × factor
  3. Always include the centroid itself
  4. Run each new θ under current architecture (whatever's in msm_base.yaml)
  5. Append results to lhs_results.jsonl with label "pert_NNN"

This is more sample-efficient than random LHS because we exploit the
strong A-monotonicity signal: all top θ have A≥0.9. We sample dense within
that region, sparse outside.

Usage:
    python scripts/perturb_around_best.py \\
        --top-k 8 --per-cluster 4 --noise-factor 2.0 \\
        --steps 50 --households 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from src.analysis.fred_targets import get_target_moments
from src.analysis.moments import moments_from_sim_csv
from src.analysis.msm import J, deep_merge, theta_to_config_patch
from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config

BOUNDS = {
    "phi_pi":     (0.5, 2.5),
    "phi_u":      (0.0, 1.2),
    "match_eff":  (0.35, 1.0),
    "price_adj":  (0.10, 1.0),
    "separation": (0.0, 0.15),
}


def cluster_topk(records: list, top_k: int) -> tuple[list, list]:
    """Split top-K records into 2 clusters by price_adj (sticky vs flexible)."""
    top = records[:top_k]
    sticky = [r for r in top if r["theta"]["price_adj"] < 0.4]
    flex = [r for r in top if r["theta"]["price_adj"] >= 0.4]
    return sticky, flex


def cluster_stats(cluster: list) -> tuple[dict, dict]:
    if not cluster:
        return {}, {}
    keys = ["phi_pi", "phi_u", "match_eff", "price_adj", "separation"]
    centroid = {k: float(np.mean([r["theta"][k] for r in cluster])) for k in keys}
    spread = {k: max(float(np.std([r["theta"][k] for r in cluster])), 0.02) for k in keys}
    return centroid, spread


def sample_around(centroid: dict, spread: dict, n: int, noise_factor: float, rng) -> list[dict]:
    samples = []
    keys = list(centroid.keys())
    for _ in range(n):
        theta = {}
        for k in keys:
            v = rng.normal(centroid[k], spread[k] * noise_factor)
            lo, hi = BOUNDS[k]
            theta[k] = float(np.clip(v, lo, hi))
        samples.append(theta)
    return samples


def override_hh(cfg: dict, n: int) -> dict:
    cfg = deepcopy(cfg)
    for g in cfg.get("agents", {}).values():
        if g.get("type") == "household":
            g["count"] = n
            break
    return cfg


async def _run_once(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


async def main_async(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config(args.base)
    target = get_target_moments()

    print(f"Target moments (live FRED): {target}")
    print(f"Architecture: history_window={base_cfg['simulation']['history_window']}, "
          f"goods_mode={base_cfg['market']['goods_clearing_mode']}")

    # Load existing data, rescore under live FRED, sort
    with open("data/msm/lhs_results.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    rescored = sorted(
        [{**r, "J": J(r["sim_moments"], target)[0]} for r in records],
        key=lambda r: r["J"],
    )

    sticky, flex = cluster_topk(rescored, args.top_k)
    print(f"\nClusters from top-{args.top_k}:")
    print(f"  Sticky: n={len(sticky)}")
    print(f"  Flex:   n={len(flex)}")

    sticky_c, sticky_s = cluster_stats(sticky)
    flex_c, flex_s = cluster_stats(flex)

    rng = np.random.default_rng(args.seed)
    candidates = []
    if sticky:
        print(f"\nSticky centroid: {sticky_c}")
        # Include centroid
        candidates.append(("pert_sticky_centroid", sticky_c))
        # Perturbed samples
        for i, theta in enumerate(sample_around(sticky_c, sticky_s, args.per_cluster, args.noise_factor, rng)):
            candidates.append((f"pert_sticky_{i:02d}", theta))
    if flex:
        print(f"Flex centroid:   {flex_c}")
        candidates.append(("pert_flex_centroid", flex_c))
        for i, theta in enumerate(sample_around(flex_c, flex_s, args.per_cluster, args.noise_factor, rng)):
            candidates.append((f"pert_flex_{i:02d}", theta))

    print(f"\nTotal candidates to evaluate: {len(candidates)}")

    # Skip already-done
    done_labels = {r["label"] for r in records}
    to_run = [(l, t) for l, t in candidates if l not in done_labels]
    print(f"To run (skipping {len(candidates) - len(to_run)} done): {len(to_run)}")

    lhs_path = Path("data/msm/lhs_results.jsonl")
    for label, theta in to_run:
        cfg = deep_merge(base_cfg, theta_to_config_patch(theta))
        cfg = override_hh(cfg, args.households)
        cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"msm_{label}"}
        t0 = time.time()
        print(f"\n=== {label}  θ = {theta} ===")
        try:
            df = await _run_once(cfg, args.steps)
            sim_m = moments_from_sim_csv(df)
            j_val, parts = J(sim_m, target)
            rec = {
                "theta": theta, "J": j_val, "parts": parts,
                "sim_moments": sim_m, "target": target,
                "elapsed_s": round(time.time() - t0, 1),
                "n_steps": args.steps, "n_households": args.households,
                "label": label,
            }
            with open(lhs_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"  J = {j_val:.2f}  unrate={sim_m.get('unrate_mean', 0):.3f}  okun={sim_m.get('okun_corr', 0):+.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            with open("data/msm/failures.jsonl", "a") as f:
                f.write(json.dumps({"label": label, "theta": theta, "error": str(e)}) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="config/experiments/msm_base.yaml")
    p.add_argument("--top-k", type=int, default=8, help="Top-K to cluster")
    p.add_argument("--per-cluster", type=int, default=4, help="Perturbed samples per cluster")
    p.add_argument("--noise-factor", type=float, default=2.0, help="Spread multiplier")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--households", type=int, default=20)
    p.add_argument("--out", default="data/msm")
    p.add_argument("--seed", type=int, default=44)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
