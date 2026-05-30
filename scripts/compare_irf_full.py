"""Full IRF comparison: both shocks × both benchmarks (NK-DSGE and RBC), 6 variables each.

Generates 4 figures (2×3 panel grids):
  fig_irf_demand_vs_dsge.png   — demand shock vs NK-DSGE
  fig_irf_demand_vs_rbc.png    — demand shock vs RBC
  fig_irf_prod_vs_dsge.png     — productivity shock vs NK-DSGE
  fig_irf_prod_vs_rbc.png      — productivity shock vs RBC
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.dsge_baselines import nk_dsge_irf, rbc_irf, smoothness

HERO_DIR = Path(os.environ.get("HERO_DATA_DIR", "data/hero_queue"))
OUT_DIR = Path("charts/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SHOCK_STEP = 10
HORIZON = 30
N_BOOT = 2000

VARS_ALL = [
    ("Output",        "total_sales"),
    ("Consumption",   "total_consumption"),
    ("Inflation",     "inflation_rate"),
    ("Unemployment",  "unemployment_rate"),
    ("Interest rate", "interest_rate"),
    ("Real wage",     "real_wage"),
]

# RBC is a real model — no nominal variables
VARS_RBC = [
    ("Output",        "total_sales"),
    ("Consumption",   "total_consumption"),
    ("Unemployment",  "unemployment_rate"),
    ("Real wage",     "real_wage"),
]

DSGE_SIGNS = {
    "demand": {
        "total_sales": +1, "total_consumption": +1, "inflation_rate": +1,
        "unemployment_rate": -1, "interest_rate": +1, "real_wage": 0,
    },
    "productivity": {
        "total_sales": +1, "total_consumption": +1, "inflation_rate": -1,
        "unemployment_rate": -1, "interest_rate": -1, "real_wage": +1,
    },
}


def load_runs(scenario: str) -> list[pd.DataFrame]:
    return [pd.read_csv(f) for f in sorted(HERO_DIR.glob(f"{scenario}_run_*.csv"))]


def bootstrap_irf(shocked: list, baseline: list, var: str, seed: int = 42):
    rng = np.random.default_rng(seed)
    S = np.stack([df[var].values for df in shocked])
    B = np.stack([df[var].values for df in baseline])
    obs = S.mean(0) - B.mean(0)
    boots = np.array([
        S[rng.integers(len(S), size=len(S))].mean(0) -
        B[rng.integers(len(B), size=len(B))].mean(0)
        for _ in range(N_BOOT)
    ])
    return obs, np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)


def peak_norm(arr):
    p = np.max(np.abs(arr))
    return arr / p if p > 1e-9 else arr


def make_figure(shock: str, benchmark: str, ref: dict, ref_label: str, ref_color: str,
                vars_list=None):
    if vars_list is None:
        vars_list = VARS_ALL
    baseline = load_runs("baseline")
    shocked = load_runs(f"{shock}_shock")

    signs = DSGE_SIGNS[shock]
    n = len(vars_list)
    ncols = 3 if n > 4 else 2
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))

    for ax, (title, var) in zip(axes.flat, vars_list):
        obs, lo, hi = bootstrap_irf(shocked, baseline, var)
        post = slice(SHOCK_STEP, SHOCK_STEP + HORIZON)

        y_ours = obs[post][:HORIZON]
        y_lo   = lo[post][:HORIZON]
        y_hi   = hi[post][:HORIZON]

        ref_key = {
            "total_sales": "output",
            "total_consumption": "consumption",
            "inflation_rate": "inflation_rate",
            "unemployment_rate": "unemployment_rate",
            "interest_rate": "interest_rate",
            "real_wage": "real_wage",
        }[var]
        ref_y = ref.get(ref_key, np.zeros(HORIZON))[:HORIZON]

        y_ours_n = peak_norm(y_ours)
        y_lo_n   = y_lo / (np.max(np.abs(y_ours)) if np.max(np.abs(y_ours)) > 1e-9 else 1)
        y_hi_n   = y_hi / (np.max(np.abs(y_ours)) if np.max(np.abs(y_ours)) > 1e-9 else 1)
        ref_n    = peak_norm(ref_y)

        x = np.arange(HORIZON)
        ax.plot(x, y_ours_n, color="#1f77b4", lw=2.2, label="LLM-ABM (ours)")
        ax.fill_between(x, y_lo_n, y_hi_n, alpha=0.18, color="#1f77b4")
        ax.plot(x, ref_n, color=ref_color, lw=1.8, ls="--", label=ref_label)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5, ls=":")

        # sign annotation
        exp_sign = signs.get(var, 0)
        sim_sign = np.sign(y_ours.mean()) if abs(y_ours.mean()) > 1e-9 else 0
        if exp_sign == 0:
            sign_str = "~"
        elif sim_sign == exp_sign:
            sign_str = "✓"
        else:
            sign_str = "✗"

        significant = not (y_lo[SHOCK_STEP - SHOCK_STEP] <= 0 <= y_hi[SHOCK_STEP - SHOCK_STEP])
        # check at t=0 post shock (i.e. SHOCK_STEP in original)
        sig_at_shock = not (lo[SHOCK_STEP] <= 0 <= hi[SHOCK_STEP])
        sig_str = "  sig✓" if sig_at_shock else ""

        ax.set_title(f"{title}  {sign_str}{sig_str}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Steps after shock", fontsize=8)
        ax.set_ylabel("Norm. to peak=±1", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    shock_label = "Demand" if shock == "demand" else "Productivity"
    fig.suptitle(
        f"{shock_label} shock IRF — LLM-ABM vs {ref_label} (peak-normalized, 95% CI)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fname = f"fig_irf_{shock}_vs_{benchmark}.png"
    out = OUT_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def main():
    nk_demand = nk_dsge_irf("demand", shock_size=1.0, horizon=60)
    nk_supply = nk_dsge_irf("supply", shock_size=1.0, horizon=60)
    rbc_prod   = rbc_irf(shock_size=0.05, horizon=60)

    # RBC has no demand-shock mechanism → all zeros (shows the gap)
    rbc_demand = {k: np.zeros(60) for k in rbc_prod}

    make_figure("demand",       "dsge", nk_demand, "NK-DSGE (Galí 2015)", "#d62728", VARS_ALL)
    make_figure("productivity", "dsge", nk_supply,  "NK-DSGE (Galí 2015)", "#d62728", VARS_ALL)
    make_figure("productivity", "rbc",  rbc_prod,   "RBC (Hansen 1985)",   "#2ca02c", VARS_RBC)


if __name__ == "__main__":
    main()
