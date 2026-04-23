"""
FRED-based WorldState initialization.

Fetches recent US macro data to calibrate the simulation's starting point
to real-world conditions. Falls back to hardcoded defaults if fredapi
is not installed or FRED_API_KEY is not set.

Approximate US values used as fallback (2023–2024 FRED data):
  - CPI inflation (annual):    ~3.4%  → quarterly ~0.85%
  - Unemployment rate:         ~3.7%
  - Fed funds rate:            ~5.25%
  - Real hourly wage index:    normalized to 25.0 (simulation units)
  - Price level:               normalized to 1.0  (simulation units)

Usage in engine or run_experiment.py:
    from src.utils.fred_init import get_initial_conditions
    init = get_initial_conditions()
    # then pass init["avg_wage"], init["unemployment_rate"], etc. to WorldState
"""

from __future__ import annotations
import os

# ---------------------------------------------------------------------------
# Hardcoded fallback (approximate US 2023–2024, FRED series)
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "avg_price":         10.0,   # normalized price level (simulation units)
    "avg_wage":          25.0,   # nominal wage per unit labor (simulation units)
    "unemployment_rate": 0.037,  # UNRATE: ~3.7%
    "interest_rate":     0.053,  # FEDFUNDS: ~5.25% → quarterly ~1.3%, use 5.3% annual
    "inflation_rate":    0.0085, # quarterly CPI inflation (~3.4% / 4)
}


def get_initial_conditions(use_fred: bool = True) -> dict:
    """
    Return a dict with macro initial conditions for WorldState.

    If fredapi is installed and FRED_API_KEY is set, fetches the latest
    available monthly observations. Otherwise returns hardcoded defaults.

    Keys: avg_price, avg_wage, unemployment_rate, interest_rate, inflation_rate
    """
    if use_fred:
        try:
            result = _fetch_from_fred()
            if result:
                print(f"[FRED] Initialized macro conditions from FRED API:")
                for k, v in result.items():
                    print(f"       {k}: {v}")
                return result
        except Exception as e:
            print(f"[FRED] Could not fetch data ({e}). Using hardcoded defaults.")

    print("[FRED] Using hardcoded US macro defaults (2023–2024).")
    return dict(_DEFAULTS)


def _fetch_from_fred() -> dict | None:
    try:
        from fredapi import Fred  # type: ignore
    except ImportError:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError("FRED_API_KEY not set in environment.")

    fred = Fred(api_key=api_key)
    result = dict(_DEFAULTS)  # start from defaults, override with live data

    # Unemployment rate
    try:
        s = fred.get_series("UNRATE", observation_start="2023-01-01")
        result["unemployment_rate"] = round(float(s.dropna().iloc[-1]) / 100, 4)
    except Exception:
        pass

    # Fed funds rate
    try:
        s = fred.get_series("FEDFUNDS", observation_start="2023-01-01")
        result["interest_rate"] = round(float(s.dropna().iloc[-1]) / 100, 4)
    except Exception:
        pass

    # CPI — quarterly inflation rate
    try:
        s = fred.get_series("CPIAUCSL", observation_start="2023-01-01").dropna()
        if len(s) >= 4:
            # approx quarterly: compare last obs to 3 months ago
            qtrly = (s.iloc[-1] - s.iloc[-4]) / s.iloc[-4]
            result["inflation_rate"] = round(float(qtrly), 6)
    except Exception:
        pass

    return result


def apply_to_world_state(world_state, conditions: dict | None = None):
    """
    Apply initial conditions to an existing WorldState object in place.
    Useful to call after WorldState() construction but before the first step.
    """
    if conditions is None:
        conditions = get_initial_conditions()

    world_state.avg_price         = conditions.get("avg_price",         world_state.avg_price)
    world_state.avg_wage          = conditions.get("avg_wage",          world_state.avg_wage)
    world_state.unemployment_rate = conditions.get("unemployment_rate", world_state.unemployment_rate)
    world_state.interest_rate     = conditions.get("interest_rate",     world_state.interest_rate)
    world_state.prev_avg_price    = world_state.avg_price  # so step-1 inflation isn't artifactual
