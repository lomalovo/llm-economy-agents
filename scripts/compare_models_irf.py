"""Compare IRF across DeepSeek-V3, DeepSeek-V4-flash, and Qwen3-235B-tput.

Shared baseline: data/hero/ (3 runs, DeepSeek-V3).
Demand shock runs per model:
  V3:        data/hero/          (3 runs)
  V4-flash:  data/hero_deepseek/ (10 runs)
  Qwen:      data/hero_qwen/     (10 runs)
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── config ──────────────────────────────────────────────────────────────────
BASELINE_DIR   = Path("data/hero")
SHOCK_DIRS     = {
    "DeepSeek V3\n(3 runs, Eliza)":        Path("data/hero"),
    "DeepSeek V4-flash\n(10 runs, official API)": Path("data/hero_deepseek"),
    "Qwen3-235B-tput\n(10 runs, Together)":  Path("data/hero_qwen"),
}
COLORS = ["#E91E63", "#2196F3", "#4CAF50"]

SHOCK_STEP = 10
N_BOOT     = 2000
SEED       = 42
OUT        = Path("charts/v2/fig_irf_model_comparison.png")

IRF_VARS = [
    ("total_consumption", "Consumption"),
    ("unemployment_rate", "Unemployment rate"),
    ("inflation_rate",    "Inflation rate"),
    ("interest_rate",     "Interest rate (CB)"),
    ("real_wage",         "Real wage"),
    ("total_sales",       "Output (total sales)"),
]

DSGE_SIGN = {
    "total_consumption": "+",
    "unemployment_rate": "-",
    "inflation_rate":    "+",
    "interest_rate":     "+",
    "real_wage":         "~",
    "total_sales":       "+",
}
# ────────────────────────────────────────────────────────────────────────────


def load_runs(pattern_dir: Path, scenario: str) -> list[pd.DataFrame]:
    files = sorted(pattern_dir.glob(f"{scenario}_run_*.csv"))
    return [pd.read_csv(f) for f in files]


def bootstrap_irf(baseline: list, shocked: list, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    steps = baseline[0]["step"].values
    out = {}
    for var, _ in IRF_VARS:
        B = np.array([df[var].values for df in baseline if var in df.columns])
        S = np.array([df[var].values for df in shocked  if var in df.columns])
        if not B.size or not S.size:
            continue
        mean_irf = S.mean(0) - B.mean(0)
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, B.shape[0], B.shape[0])
            s = rng.integers(0, S.shape[0], S.shape[0])
            boots.append(S[s].mean(0) - B[b].mean(0))
        boots = np.array(boots)
        out[var] = (steps, mean_irf,
                    np.percentile(boots, 2.5, 0),
                    np.percentile(boots, 97.5, 0))
    return out


def main():
    baseline_runs = load_runs(BASELINE_DIR, "baseline")
    print(f"Baseline runs: {len(baseline_runs)}")

    # Compute IRF per model
    model_irfs = {}
    for label, shock_dir in SHOCK_DIRS.items():
        runs = load_runs(shock_dir, "demand_shock")
        print(f"{label.split(chr(10))[0]}: {len(runs)} shock runs")
        model_irfs[label] = bootstrap_irf(baseline_runs, runs)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    fig.suptitle(
        "Demand Shock IRF — Model Comparison\n"
        "DeepSeek V3 (3 runs)  vs  DeepSeek V4-flash (10 runs)  vs  Qwen3-235B-tput (10 runs)\n"
        "Shared baseline: data/hero (3 runs, DeepSeek V3). Bootstrap 95% CI, B=2000.",
        fontsize=10, fontweight="bold",
    )

    for ax, (var, label) in zip(axes, IRF_VARS):
        dsge = DSGE_SIGN.get(var, "~")
        for (model_label, irf_dict), color in zip(model_irfs.items(), COLORS):
            if var not in irf_dict:
                continue
            steps, mean_irf, lo, hi = irf_dict[var]
            short = model_label.split("\n")[0]
            ax.plot(steps, mean_irf, color=color, linewidth=2, label=short)
            ax.fill_between(steps, lo, hi, alpha=0.15, color=color)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.axvline(SHOCK_STEP, color="red", linewidth=1.0, linestyle=":",
                   label=f"Shock (step {SHOCK_STEP})")
        ax.set_title(f"{label}  |  DSGE: {dsge}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Step", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUT}")

    # ── print significance table ──────────────────────────────────────────────
    print(f"\n{'Variable':<22} {'Model':<22} {'IRF@10':>8} {'95% CI':>22} {'Sig':>5}")
    print("-" * 85)
    for var, vlabel in IRF_VARS:
        for model_label, irf_dict in model_irfs.items():
            if var not in irf_dict:
                continue
            steps, mean_irf, lo, hi = irf_dict[var]
            idx = np.searchsorted(steps, SHOCK_STEP)
            post = steps >= SHOCK_STEP
            sig = "YES" if (np.any(lo[post] > 0) or np.any(hi[post] < 0)) else "no"
            short = model_label.split("\n")[0]
            print(f"{vlabel:<22} {short:<22} {mean_irf[idx]:>+8.4f} "
                  f"[{lo[idx]:>+8.4f}, {hi[idx]:>+8.4f}] {sig:>5}")
        print()


if __name__ == "__main__":
    main()
