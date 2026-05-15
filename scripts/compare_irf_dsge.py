"""Side-by-side comparison: our LLM-ABM hero IRFs vs. analytical NK-DSGE / RBC.

Loads:
  - data/hero/{baseline,demand_shock,productivity_shock}_run_*.csv → bootstrap IRF
  - dsge_baselines.nk_dsge_irf() and rbc_irf()

Produces:
  - charts/v2/fig_irf_vs_dsge_demand.png    — 4-panel grid (Y, π, u, r)
  - charts/v2/fig_irf_vs_rbc_productivity.png — 4-panel grid (Y, C, I, u)
  - data/irf_comparison_metrics.json         — smoothness ratios + sign agreement

All IRFs are normalized to compare directionality and shape, not absolute level.
LLM-ABM is in simulation units; NK/RBC are log-deviations. We re-scale to %
of pre-shock level so they sit on the same axis.
"""
from __future__ import annotations

import json
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


HERO_DIR = Path(os.environ.get("HERO_DATA_DIR", "data/hero"))
OUT_DIR = Path("charts/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOCK_STEP = 10


def load_hero(scenario: str) -> list[pd.DataFrame]:
    return [pd.read_csv(f) for f in sorted(HERO_DIR.glob(f"{scenario}_run_*.csv"))]


def bootstrap_irf(B: np.ndarray, S: np.ndarray, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    mean_irf = S.mean(axis=0) - B.mean(axis=0)
    boots = []
    for _ in range(n_boot):
        bi = rng.integers(0, B.shape[0], B.shape[0])
        si = rng.integers(0, S.shape[0], S.shape[0])
        boots.append(S[si].mean(axis=0) - B[bi].mean(axis=0))
    boots = np.array(boots)
    return mean_irf, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def normalize_to_pct(series: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Convert level deviation to percent of pre-shock baseline."""
    base = baseline[:SHOCK_STEP].mean() if len(baseline) >= SHOCK_STEP else baseline.mean()
    if abs(base) < 1e-9:
        return series
    return 100.0 * series / abs(base)


def compute_hero_irf(scenario: str, var: str) -> dict:
    B = np.stack([df[var].values for df in load_hero("baseline")], axis=0)
    S = np.stack([df[var].values for df in load_hero(scenario)], axis=0)
    horizon = B.shape[1]
    mean_irf, lo, hi = bootstrap_irf(B, S)
    baseline_mean = B.mean(axis=0)
    return {
        "horizon": horizon,
        "mean": mean_irf,
        "lo": lo,
        "hi": hi,
        "baseline_level": baseline_mean,
        "post_steps": np.arange(SHOCK_STEP, horizon) - SHOCK_STEP,
    }


def _peak_normalize(arr: np.ndarray) -> np.ndarray:
    """Scale to peak=1 by abs value, preserving sign."""
    peak = np.max(np.abs(arr))
    return arr / peak if peak > 1e-9 else arr


def plot_demand_comparison():
    """4-panel: Output, Inflation, Unemployment, Interest rate.

    Plots peak-normalized IRFs to compare SHAPE (not magnitude).
    Smoothness metric on peak-normalized series — fair comparison.
    Reports both peak agreement direction + half-life persistence.
    """
    nk = nk_dsge_irf("demand", shock_size=1.0, horizon=30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("Output (total_sales)", "total_sales", "output"),
        ("Inflation rate", "inflation_rate", "inflation_rate"),
        ("Unemployment rate", "unemployment_rate", "unemployment_rate"),
        ("Interest rate", "interest_rate", "interest_rate"),
    ]
    metrics = {}
    for ax, (title, hero_var, dsge_var) in zip(axes.flat, panels):
        h = compute_hero_irf("demand_shock", hero_var)
        post = slice(SHOCK_STEP, SHOCK_STEP + 30)
        x_ours = np.arange(30)
        y_ours_raw = h["mean"][post][:30]
        y_lo_raw = h["lo"][post][:30]
        y_hi_raw = h["hi"][post][:30]
        dsge_y_raw = nk[dsge_var][:30]

        # Peak-normalize each series independently to compare SHAPES
        peak_ours = np.max(np.abs(y_ours_raw)) if np.max(np.abs(y_ours_raw)) > 1e-9 else 1.0
        peak_dsge = np.max(np.abs(dsge_y_raw)) if np.max(np.abs(dsge_y_raw)) > 1e-9 else 1.0
        y_ours_n = y_ours_raw / peak_ours
        y_lo_n = y_lo_raw / peak_ours
        y_hi_n = y_hi_raw / peak_ours
        dsge_y_n = dsge_y_raw / peak_dsge

        s_ours = smoothness(y_ours_n)
        s_dsge = smoothness(dsge_y_n)
        metrics[hero_var] = {
            "ours_smoothness_normalized": s_ours,
            "dsge_smoothness_normalized": s_dsge,
            "ours_peak_raw": float(peak_ours),
            "dsge_peak_raw": float(peak_dsge),
            "ours_peak_timing": int(np.argmax(np.abs(y_ours_raw))),
            "dsge_peak_timing": int(np.argmax(np.abs(dsge_y_raw))),
            "sign_agree": bool(np.sign(y_ours_raw.mean()) == np.sign(dsge_y_raw.mean())),
        }

        ax.plot(x_ours, y_ours_n, color="#1f77b4", linewidth=2.4, label="LLM-ABM (ours)")
        ax.fill_between(x_ours, y_lo_n, y_hi_n, alpha=0.18, color="#1f77b4")
        ax.plot(x_ours, dsge_y_n, color="#d62728", linewidth=1.8, linestyle="--", label="NK-DSGE")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(
            f"{title}\n"
            f"peak: ours={peak_ours:.3f} dsge={peak_dsge:.3f} | "
            f"shape-smoothness: ours={s_ours:.2f} dsge={s_dsge:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Steps after shock")
        ax.set_ylabel("Normalized to peak=±1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Demand shock IRF (peak-normalized): LLM-ABM vs. NK-DSGE", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig_irf_vs_dsge_demand.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    return metrics


def plot_productivity_comparison():
    """4-panel: Output, Consumption, Unemployment, Real wage. Peak-normalized."""
    rbc = rbc_irf(shock_size=0.05, horizon=30)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("Output (total_sales)", "total_sales", "output"),
        ("Total consumption", "total_consumption", "consumption"),
        ("Unemployment rate", "unemployment_rate", "unemployment_rate"),
        ("Real wage", "real_wage", "real_wage"),
    ]
    metrics = {}
    for ax, (title, hero_var, rbc_var) in zip(axes.flat, panels):
        h = compute_hero_irf("productivity_shock", hero_var)
        post = slice(SHOCK_STEP, SHOCK_STEP + 30)
        x_ours = np.arange(30)
        y_ours_raw = h["mean"][post][:30]
        y_lo_raw = h["lo"][post][:30]
        y_hi_raw = h["hi"][post][:30]
        rbc_y_raw = rbc[rbc_var][:30]

        peak_ours = np.max(np.abs(y_ours_raw)) if np.max(np.abs(y_ours_raw)) > 1e-9 else 1.0
        peak_rbc = np.max(np.abs(rbc_y_raw)) if np.max(np.abs(rbc_y_raw)) > 1e-9 else 1.0
        y_ours_n = y_ours_raw / peak_ours
        y_lo_n = y_lo_raw / peak_ours
        y_hi_n = y_hi_raw / peak_ours
        rbc_y_n = rbc_y_raw / peak_rbc

        s_ours = smoothness(y_ours_n)
        s_rbc = smoothness(rbc_y_n)
        metrics[hero_var] = {
            "ours_smoothness_normalized": s_ours,
            "rbc_smoothness_normalized": s_rbc,
            "ours_peak_raw": float(peak_ours),
            "rbc_peak_raw": float(peak_rbc),
            "ours_peak_timing": int(np.argmax(np.abs(y_ours_raw))),
            "rbc_peak_timing": int(np.argmax(np.abs(rbc_y_raw))),
            "sign_agree": bool(np.sign(y_ours_raw.mean()) == np.sign(rbc_y_raw.mean())),
        }

        ax.plot(x_ours, y_ours_n, color="#1f77b4", linewidth=2.4, label="LLM-ABM (ours)")
        ax.fill_between(x_ours, y_lo_n, y_hi_n, alpha=0.18, color="#1f77b4")
        ax.plot(x_ours, rbc_y_n, color="#2ca02c", linewidth=1.8, linestyle="--", label="RBC (Hansen 1985)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_title(
            f"{title}\n"
            f"peak: ours={peak_ours:.3f} rbc={peak_rbc:.3f} | "
            f"shape-smoothness: ours={s_ours:.2f} rbc={s_rbc:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("Steps after shock")
        ax.set_ylabel("Normalized to peak=±1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Productivity shock IRF (peak-normalized): LLM-ABM vs. RBC", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig_irf_vs_rbc_productivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    return metrics


def main():
    metrics = {"demand_vs_nk": plot_demand_comparison(),
               "productivity_vs_rbc": plot_productivity_comparison()}
    out = Path("data/irf_comparison_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"\nSaved metrics to {out}")

    print("\n=== Shape comparison (peak-normalized smoothness) ===")
    print("\nDemand shock (vs NK-DSGE):")
    print(f"  {'var':<22} {'sign':>6} {'shape_smooth_ours':>20} {'_dsge':>10} {'peak_t_ours':>14} {'_dsge':>8}")
    for var, m in metrics["demand_vs_nk"].items():
        sign = "✓" if m["sign_agree"] else "✗"
        print(f"  {var:<22} {sign:>6}  {m['ours_smoothness_normalized']:>18.3f}  {m['dsge_smoothness_normalized']:>8.3f}  {m['ours_peak_timing']:>12d}  {m['dsge_peak_timing']:>6d}")
    print("\nProductivity shock (vs RBC):")
    print(f"  {'var':<22} {'sign':>6} {'shape_smooth_ours':>20} {'_rbc':>10} {'peak_t_ours':>14} {'_rbc':>8}")
    for var, m in metrics["productivity_vs_rbc"].items():
        sign = "✓" if m["sign_agree"] else "✗"
        print(f"  {var:<22} {sign:>6}  {m['ours_smoothness_normalized']:>18.3f}  {m['rbc_smoothness_normalized']:>8.3f}  {m['ours_peak_timing']:>12d}  {m['rbc_peak_timing']:>6d}")


if __name__ == "__main__":
    main()
