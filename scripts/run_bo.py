"""Bayesian Optimization on top of existing LHS results.

Loads data/msm/lhs_results.jsonl, fits a Gaussian Process surrogate on
(theta_i, J_i) pairs, and proposes new theta values via Expected
Improvement (EI). Runs each proposal, appends result to lhs_results.jsonl
with label 'bo_NNN'.

Usage:
    python scripts/run_bo.py --n-iter 5 --steps 50 --households 20
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
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

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
PARAM_NAMES = list(BOUNDS.keys())


def theta_to_x(theta: dict) -> np.ndarray:
    return np.array([
        (theta[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0])
        for k in PARAM_NAMES
    ])


def x_to_theta(x: np.ndarray) -> dict:
    return {
        k: float(BOUNDS[k][0] + x[i] * (BOUNDS[k][1] - BOUNDS[k][0]))
        for i, k in enumerate(PARAM_NAMES)
    }


def load_lhs(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    X = np.array([theta_to_x(r["theta"]) for r in records])
    y = np.array([r["J"] for r in records])
    # Cap J at 1500 to prevent runaway records from dominating GP fit
    y = np.minimum(y, 1500.0)
    return X, y, records


def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    log_y = np.log(y + 1.0)
    kernel = (
        C(1.0, (1e-2, 1e2))
        * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 1e1), nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-4, 1e0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=42
    )
    gp.fit(X, log_y)
    return gp


def expected_improvement(x: np.ndarray, gp: GaussianProcessRegressor, y_min: float) -> float:
    x = x.reshape(1, -1)
    mu, sigma = gp.predict(x, return_std=True)
    sigma = max(float(sigma[0]), 1e-9)
    mu = float(mu[0])
    z = (y_min - mu) / sigma
    ei = (y_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    return float(ei)


def propose_next(gp, y_min: float, X_existing: np.ndarray, n_candidates: int = 2000) -> np.ndarray:
    rng = np.random.default_rng()
    cands = rng.uniform(0, 1, size=(n_candidates, len(PARAM_NAMES)))
    eis = np.array([expected_improvement(c, gp, y_min) for c in cands])
    min_dist = np.array([np.min(np.linalg.norm(X_existing - c, axis=1)) for c in cands])
    # Require some minimum separation from already-evaluated points
    eis_adjusted = eis * (min_dist > 0.08)
    if np.all(eis_adjusted <= 0):
        return cands[np.argmax(eis)]
    return cands[np.argmax(eis_adjusted)]


async def _run_once(cfg: dict, steps: int) -> pd.DataFrame:
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


async def main_async(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lhs_path = out_dir / "lhs_results.jsonl"

    base_cfg = load_config(args.base)
    target = get_target_moments()

    X, y, all_records = load_lhs(lhs_path)
    print(f"Loaded {len(X)} existing points. Best J so far = {y.min():.2f}")

    n_bo_done = sum(1 for r in all_records if r["label"].startswith("bo_"))

    for it in range(args.n_iter):
        gp = fit_gp(X, y)
        y_min = float(np.log(y.min() + 1.0))
        next_x = propose_next(gp, y_min, X)
        theta_next = x_to_theta(next_x)
        label = f"bo_{n_bo_done + it:03d}"
        print(f"\n=== BO iter {it + 1}/{args.n_iter}  label={label} ===")
        print(f"  θ = {theta_next}")
        mu, sd = gp.predict(next_x.reshape(1, -1), return_std=True)
        print(f"  GP predicts log(J+1)={mu[0]:.2f} ± {sd[0]:.2f}")

        cfg = deep_merge(base_cfg, theta_to_config_patch(theta_next))
        cfg = override_hh(cfg, args.households)
        cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"msm_{label}"}
        try:
            df = await _run_once(cfg, args.steps)
            sim_m = moments_from_sim_csv(df)
            j_val, parts = J(sim_m, target)
            rec = {
                "theta": theta_next, "J": j_val, "parts": parts,
                "sim_moments": sim_m, "target": target,
                "elapsed_s": None, "n_steps": args.steps,
                "n_households": args.households, "label": label,
            }
            with open(lhs_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"  ACTUAL J = {j_val:.2f}  (unrate={sim_m.get('unrate_mean', 0):.3f}, okun={sim_m.get('okun_corr', 0):+.3f})")
            X = np.vstack([X, next_x])
            y = np.append(y, min(j_val, 1500.0))
            print(f"  Best J overall = {y.min():.2f}")
        except Exception as e:
            print(f"  FAILED: {e}")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="config/experiments/msm_base.yaml")
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--households", type=int, default=20)
    p.add_argument("--out", default="data/msm")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
