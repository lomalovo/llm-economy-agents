"""Method of Simulated Moments calibration.

J(θ) = Σ_k w_k · ((m_sim_k(θ) − m_target_k) / scale_k)^2

Default weights prioritize the moments most widely used to assess macroeconomic
realism: unemployment level + persistence, inflation volatility + persistence,
output persistence, and the three cross-correlations (Okun, Phillips, Beveridge).
Scales normalize different-magnitude moments onto a common axis so J is not
dominated by large-mean moments.
"""
from __future__ import annotations

import math
import numpy as np


# Moments used in J. Keep this aligned with compute_moments() output keys.
CALIBRATION_MOMENTS = (
    "unrate_mean",
    "unrate_ar1",
    "inflation_std",
    "inflation_ar1",
    "output_ar1",
    "okun_corr",
    "phillips_corr",
    "beveridge_corr",
)

# Per-moment scale: roughly the typical magnitude of the moment in data.
# J contribution of each moment is bounded O(1) when sim matches target.
DEFAULT_SCALES = {
    "unrate_mean":    0.03,     # ~3pp around mean
    "unrate_ar1":     0.10,
    "inflation_std":  0.01,
    "inflation_ar1":  0.20,
    "output_ar1":     0.05,
    "okun_corr":      0.30,
    "phillips_corr":  0.30,
    "beveridge_corr": 0.30,
}

DEFAULT_WEIGHTS = {
    "unrate_mean":    1.5,
    "unrate_ar1":     1.0,
    "inflation_std":  1.5,
    "inflation_ar1":  1.0,
    "output_ar1":     0.5,
    "okun_corr":      1.0,
    "phillips_corr":  1.0,
    "beveridge_corr": 1.5,    # Beveridge is the load-bearing one — MP matching is why it exists
}


def J(
    sim_moments: dict,
    target_moments: dict,
    weights: dict | None = None,
    scales: dict | None = None,
    keys: tuple = CALIBRATION_MOMENTS,
) -> tuple[float, dict]:
    """Return (J_value, per_moment_contributions). NaNs in sim_moments are
    treated as maximum contribution (penalty) so MSM steers away from params
    that produce degenerate simulations."""
    weights = weights or DEFAULT_WEIGHTS
    scales  = scales  or DEFAULT_SCALES

    parts = {}
    total = 0.0
    for k in keys:
        if k not in target_moments:
            continue
        m_sim    = sim_moments.get(k, float("nan"))
        m_target = target_moments[k]
        scale    = scales.get(k, 1.0)
        w        = weights.get(k, 1.0)
        if isinstance(m_sim, float) and math.isnan(m_sim):
            contrib = 4.0 * w  # penalty: equivalent to ~2σ miss
        else:
            contrib = w * ((m_sim - m_target) / scale) ** 2
        parts[k] = contrib
        total   += contrib
    return total, parts


def latin_hypercube(n: int, bounds: dict, seed: int = 42) -> list[dict]:
    """Generate n LHS samples in the parameter space defined by bounds={name:(lo,hi)}.

    Returns list of dicts {param_name: value}. Uniform sampling within each
    stratum, then a random permutation combines them across dimensions.
    """
    rng = np.random.default_rng(seed)
    names = list(bounds.keys())
    d = len(names)
    # Normalize samples in [0,1)
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    samples = np.empty((n, d))
    for j in range(d):
        samples[:, j] = cut[:-1] + u[:, j] * (cut[1:] - cut[:-1])
        rng.shuffle(samples[:, j])

    out = []
    for i in range(n):
        theta = {}
        for j, name in enumerate(names):
            lo, hi = bounds[name]
            theta[name] = float(lo + samples[i, j] * (hi - lo))
        out.append(theta)
    return out


def find_best(results: list[dict]) -> dict:
    """Return the result with smallest J."""
    if not results:
        raise ValueError("empty results")
    return min(results, key=lambda r: r.get("J", float("inf")))


def theta_to_config_patch(theta: dict) -> dict:
    """Map a θ dict into a nested config patch that can be deep-merged with a
    base experiment config.

    Expected parameter names (bounds defined in run_msm.py):
        phi_pi         → central_bank.inflation_sensitivity
        phi_u          → central_bank.unemployment_sensitivity
        match_eff      → market.matching_efficiency
        price_adj      → market.price_adjustment_speed
        separation     → market.separation_rate
        tax_top        → government.tax_brackets[-1].rate (optional)
    """
    patch = {"central_bank": {}, "market": {}, "government": {}}
    if "phi_pi" in theta:
        patch["central_bank"]["inflation_sensitivity"] = theta["phi_pi"]
    if "phi_u" in theta:
        patch["central_bank"]["unemployment_sensitivity"] = theta["phi_u"]
    if "match_eff" in theta:
        patch["market"]["matching_efficiency"] = theta["match_eff"]
    if "price_adj" in theta:
        patch["market"]["price_adjustment_speed"] = theta["price_adj"]
    if "separation" in theta:
        patch["market"]["separation_rate"] = theta["separation"]
    return patch


def deep_merge(base: dict, patch: dict) -> dict:
    """Deep-merge patch into a copy of base (patch wins on conflict)."""
    import copy
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
