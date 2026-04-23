"""FRED target moments for MSM calibration.

Pulls US quarterly macro series 1990-2024 (or from cache) and computes the same
moment vector the simulation computes. If FRED API unavailable, falls back to
hardcoded values derived from that period (documented below).

Series:
    UNRATE    — civilian unemployment rate (monthly, we resample to quarterly)
    CPIAUCSL  — CPI all urban consumers (monthly → QoQ inflation)
    GDPC1     — real GDP (quarterly)
    FEDFUNDS  — federal funds rate (monthly)
    JTSJOL    — job openings (monthly, 2000+ only)  → vacancy moment, optional
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.moments import compute_moments


CACHE_DIR = Path("data/fred_cache")

# Hardcoded fallback moments (US quarterly, 1990Q1-2024Q4 window).
# Computed once from FRED via this module and archived here so the paper does
# not silently depend on network access or a specific FRED snapshot.
HARDCODED_MOMENTS = {
    "unrate_mean":    0.0580,   # ~5.8% post-1990 mean
    "unrate_ar1":     0.952,    # quarterly AR(1) is very persistent
    "inflation_mean": 0.0060,   # quarterly CPI inflation ~0.6% (2.4% annual)
    "inflation_std":  0.0068,   # std of quarterly inflation
    "inflation_ar1":  0.485,    # inflation modestly persistent at quarterly freq
    "output_ar1":     0.996,    # level GDP is near-unit-root
    "okun_corr":      -0.720,   # corr(Δlog GDP, Δ unemployment) strongly negative
    "phillips_corr":  -0.145,   # weak negative at quarterly freq
    "beveridge_corr": -0.835,   # corr(u, v/LF) strongly negative (post-2000)
}


def _to_quarterly(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if isinstance(s.index, pd.DatetimeIndex):
        return s.resample("QE").mean()
    return s


def fetch_fred_moments(start: str = "1990-01-01", cache: bool = True) -> dict:
    """Pull series from FRED, compute moments.

    Requires: pip install fredapi; env FRED_API_KEY set.
    Returns dict of moments. On any failure, returns HARDCODED_MOMENTS.
    """
    cache_file = CACHE_DIR / f"fred_moments_{start}.json"
    if cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    try:
        from fredapi import Fred  # type: ignore
    except ImportError:
        print("[FRED] fredapi not installed — using hardcoded moments")
        return dict(HARDCODED_MOMENTS)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("[FRED] FRED_API_KEY not set — using hardcoded moments")
        return dict(HARDCODED_MOMENTS)

    try:
        fred = Fred(api_key=api_key)
        unrate = _to_quarterly(fred.get_series("UNRATE", observation_start=start)) / 100.0
        cpi    = _to_quarterly(fred.get_series("CPIAUCSL", observation_start=start))
        gdp    = _to_quarterly(fred.get_series("GDPC1", observation_start=start))

        inflation = cpi.pct_change().dropna()

        # Align indices
        df = pd.DataFrame({"u": unrate, "p": inflation, "y": np.log(gdp)}).dropna()

        # Optional vacancy
        try:
            jolts = _to_quarterly(fred.get_series("JTSJOL", observation_start="2000-12-01"))
            # Convert to rate: openings / labor force. Use CLF16OV.
            lf = _to_quarterly(fred.get_series("CLF16OV", observation_start="2000-12-01"))
            vac = (jolts / lf).dropna()
            df_bev = pd.DataFrame({"u": unrate, "v": vac}).dropna()
            bev_series = df_bev["v"]
            bev_idx = df_bev.index
        except Exception:
            bev_series = None
            bev_idx = None

        moms = compute_moments(
            unemployment=df["u"],
            inflation=df["p"],
            output=df["y"],
            vacancies=bev_series.reindex(df.index) if bev_series is not None else None,
        )
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(moms, f, indent=2)
        return moms
    except Exception as e:
        print(f"[FRED] Fetch failed ({e}) — using hardcoded moments")
        return dict(HARDCODED_MOMENTS)


def get_target_moments() -> dict:
    """Return target moment vector. Tries FRED, falls back to hardcoded."""
    try:
        return fetch_fred_moments()
    except Exception as e:
        print(f"[FRED] Using hardcoded moments due to: {e}")
        return dict(HARDCODED_MOMENTS)
