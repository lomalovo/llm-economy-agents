"""Moment computation for MSM calibration.

Shared between FRED (empirical targets) and sim CSV (model predictions). Keeping
the formulas identical is the whole point — if target_moments and sim_moments
are computed differently, the calibration is measuring the computation gap, not
the model gap.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ar1(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 4:
        return float("nan")
    return float(np.corrcoef(s.iloc[:-1].values, s.iloc[1:].values)[0, 1])


def _corr(a: pd.Series, b: pd.Series) -> float:
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 4 or df["a"].std() == 0 or df["b"].std() == 0:
        return float("nan")
    return float(np.corrcoef(df["a"], df["b"])[0, 1])


def compute_moments(
    unemployment: pd.Series,
    inflation: pd.Series,
    output: pd.Series,
    vacancies: pd.Series | None = None,
) -> dict:
    """Compute the canonical moment vector.

    Keys:
        unrate_mean       — mean unemployment rate (level, fraction)
        unrate_ar1        — AR(1) of unemployment
        inflation_mean    — mean period-over-period inflation (fraction)
        inflation_std     — std of inflation (fraction)
        inflation_ar1     — AR(1) of inflation
        output_ar1        — AR(1) of output (levels)
        okun_corr         — corr(Δ output, Δ unemployment)
        phillips_corr     — corr(unemployment, inflation)
        beveridge_corr    — corr(unemployment, vacancies) [if vacancies given]
    """
    u = pd.Series(unemployment).astype(float)
    p = pd.Series(inflation).astype(float)
    y = pd.Series(output).astype(float)

    du = u.diff()
    dy = y.diff()

    out = {
        "unrate_mean":    float(u.mean()),
        "unrate_ar1":     _ar1(u),
        "inflation_mean": float(p.mean()),
        "inflation_std":  float(p.std()),
        "inflation_ar1":  _ar1(p),
        "output_ar1":     _ar1(y),
        "okun_corr":      _corr(dy, du),
        "phillips_corr":  _corr(u, p),
    }
    if vacancies is not None:
        v = pd.Series(vacancies).astype(float)
        out["beveridge_corr"] = _corr(u, v)
    return out


def moments_from_sim_csv(df: pd.DataFrame, skip_transient: int = 5) -> dict:
    """Compute moments from a simulation run CSV.

    skip_transient drops the first N steps where price discovery / inventory
    build-up dominate. Default 5 matches stylized_facts.py.
    """
    d = df[df["step"] > skip_transient].reset_index(drop=True)
    return compute_moments(
        unemployment=d["unemployment_rate"],
        inflation=d["inflation_rate"],
        output=d["total_sales"],
        vacancies=d["vacancy_rate"] if "vacancy_rate" in d.columns else None,
    )


def moments_from_multiple_runs(dfs: list[pd.DataFrame], skip_transient: int = 5) -> dict:
    """Compute per-run moments then average them (reduces run-to-run noise)."""
    if not dfs:
        return {}
    all_m = [moments_from_sim_csv(df, skip_transient) for df in dfs]
    keys = all_m[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_m if not (isinstance(m[k], float) and np.isnan(m[k]))]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[k + "_std"] = float(np.std(vals)) if vals else float("nan")
    return agg
