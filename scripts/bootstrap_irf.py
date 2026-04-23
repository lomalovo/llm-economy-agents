"""
Bootstrap confidence intervals on IRF.
Uses existing avg CSVs + individual run CSVs to compute 95% CI via bootstrap resampling.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

IRF_VARS = [
    ("total_sales",       "Output (total sales)"),
    ("inflation_rate",    "Inflation rate"),
    ("unemployment_rate", "Unemployment rate"),
    ("interest_rate",     "Interest rate (CB)"),
    ("real_wage",         "Real wage"),
    ("total_consumption", "Total consumption"),
]

DSGE_DEMAND = {
    "total_sales": "+",
    "inflation_rate": "+",
    "unemployment_rate": "-",
    "interest_rate": "+",
    "real_wage": "~",
    "total_consumption": "+",
}

DSGE_PRODUCTIVITY = {
    "total_sales": "+",
    "inflation_rate": "-",
    "unemployment_rate": "-",
    "interest_rate": "-",
    "real_wage": "+",
    "total_consumption": "+",
}


def load_runs(scenario: str, data_dir: Path) -> list[pd.DataFrame]:
    """Load all individual run CSVs for a scenario."""
    runs = sorted(data_dir.glob(f"{scenario}_run_*.csv"))
    if not runs:
        raise FileNotFoundError(f"No run files found for scenario '{scenario}' in {data_dir}")
    return [pd.read_csv(r) for r in runs]


def bootstrap_irf(baseline_runs: list, shocked_runs: list, n_boot: int = 2000, seed: int = 42):
    """
    Returns dict: var -> (steps, mean_irf, lower_ci, upper_ci)
    Bootstrap by resampling pairs (baseline_i, shocked_j) with replacement.
    """
    rng = np.random.default_rng(seed)
    n_b = len(baseline_runs)
    n_s = len(shocked_runs)
    steps = baseline_runs[0]["step"].values

    results = {}
    for var, _ in IRF_VARS:
        # Stack: shape (n_runs, n_steps)
        B = np.array([df[var].values for df in baseline_runs if var in df.columns])
        S = np.array([df[var].values for df in shocked_runs if var in df.columns])
        if B.shape[0] == 0 or S.shape[0] == 0:
            continue

        # Point estimate
        mean_irf = S.mean(axis=0) - B.mean(axis=0)

        # Bootstrap
        boot_irfs = []
        for _ in range(n_boot):
            b_idx = rng.integers(0, B.shape[0], size=B.shape[0])
            s_idx = rng.integers(0, S.shape[0], size=S.shape[0])
            boot_irf = S[s_idx].mean(axis=0) - B[b_idx].mean(axis=0)
            boot_irfs.append(boot_irf)

        boot_irfs = np.array(boot_irfs)
        lower = np.percentile(boot_irfs, 2.5, axis=0)
        upper = np.percentile(boot_irfs, 97.5, axis=0)

        results[var] = (steps, mean_irf, lower, upper)

    return results


def plot_irf_ci(irf_results: dict, shock_step: int, shock_name: str,
                dsge_predictions: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    fig.suptitle(f"Impulse Response Functions — {shock_name}\n"
                 f"with 95% Bootstrap Confidence Intervals", fontsize=13, fontweight='bold')

    for ax, (var, label) in zip(axes, IRF_VARS):
        if var not in irf_results:
            ax.set_visible(False)
            continue

        steps, mean_irf, lower, upper = irf_results[var]
        dsge_sign = dsge_predictions.get(var, "~")

        # Check significance: does CI exclude 0 at any post-shock step?
        post = steps >= shock_step
        # Significant if CI excludes 0 at ANY post-shock step (peak significance)
        significant = np.any((lower[post] > 0) | (upper[post] < 0))
        correct_sign = (dsge_sign == "+" and mean_irf[post].mean() > 0) or \
                       (dsge_sign == "-" and mean_irf[post].mean() < 0) or \
                       (dsge_sign == "~")

        color = "#2196F3"
        ax.plot(steps, mean_irf, color=color, linewidth=2, label="IRF (mean)")
        ax.fill_between(steps, lower, upper, alpha=0.25, color=color, label="95% CI")
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(shock_step, color='red', linewidth=1.2, linestyle=':', label=f"Shock (step {shock_step})")

        verdict = ""
        if significant and correct_sign:
            verdict = "✓ Significant, correct sign"
            ax.set_facecolor("#f0fff0")
        elif correct_sign:
            verdict = "~ Correct sign, CI touches 0"
            ax.set_facecolor("#fffff0")
        else:
            verdict = "✗ Wrong sign"
            ax.set_facecolor("#fff0f0")

        ax.set_title(f"{label}\nDSGE: {dsge_sign}  |  {verdict}", fontsize=9)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/experiments")
    parser.add_argument("--out_dir", default="charts")
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    for shock_name, scenario, dsge_preds, shock_step in [
        ("Demand Shock (cash injection +100)", "demand_shock", DSGE_DEMAND, 10),
        ("Productivity Shock (×1.5)", "productivity_shock", DSGE_PRODUCTIVITY, 10),
    ]:
        print(f"\n=== {shock_name} ===")
        try:
            baseline_runs = load_runs("baseline", data_dir)
            shocked_runs  = load_runs(scenario, data_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Baseline runs: {len(baseline_runs)}, Shocked runs: {len(shocked_runs)}")

        irf = bootstrap_irf(baseline_runs, shocked_runs, n_boot=args.n_boot)

        # Print significance table
        print(f"\n  {'Variable':<25} {'Mean IRF(step10)':>17} {'95% CI':>22} {'Sig':>6} {'DSGE':>6}")
        print("  " + "-"*80)
        steps_arr = baseline_runs[0]["step"].values
        shock_idx = np.where(steps_arr == shock_step)[0]
        for var, label in IRF_VARS:
            if var not in irf:
                continue
            _, mean_irf, lower, upper = irf[var]
            if len(shock_idx):
                # Check significance at shock step AND at peak post-shock
                post_steps = steps_arr >= shock_step
                post_mean  = mean_irf[post_steps]
                post_lower = lower[post_steps]
                post_upper = upper[post_steps]
                # Peak = step with max |mean_irf|
                peak_rel = int(np.argmax(np.abs(post_mean)))
                sig_shock = lower[shock_idx[0]] > 0 or upper[shock_idx[0]] < 0
                sig_peak  = post_lower[peak_rel] > 0 or post_upper[peak_rel] < 0
                sig = "YES" if sig_peak else ("impact" if sig_shock else "no")
                dsge = dsge_preds.get(var, "~")
                peak_step = steps_arr[post_steps][peak_rel]
                i = shock_idx[0]
                print(f"  {label:<25} {mean_irf[i]:>+17.4f} [{lower[i]:>+9.4f}, {upper[i]:>+9.4f}] {sig:>8} {dsge:>6}  (peak step={peak_step})")

        fname = scenario.replace("_", "-")
        plot_irf_ci(irf, shock_step, shock_name, dsge_preds,
                    out_dir / f"irf_ci_{fname}.png")


if __name__ == "__main__":
    main()
