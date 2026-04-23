"""
Multi-run experiment runner for DSGE IRF validation.

Usage:
    python run_experiment.py --config config/experiments/baseline.yaml --runs 5 --steps 30
    python run_experiment.py --config config/experiments/demand_shock.yaml --runs 5 --steps 30

Outputs per scenario (saved to --out directory):
    {name}_run_{i}.csv   — individual run data
    {name}_avg.csv       — mean across runs (used for IRF computation)
    {name}_std.csv       — std across runs (used for confidence bands)
"""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.llm.factory import get_llm_backend
from src.core.engine import SimulationEngine


async def _run_single(cfg: dict, steps: int) -> list[dict]:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return engine.logger.data


async def run_experiment(config_path: str, n_runs: int, steps: int, out_dir: str):
    cfg = load_config(config_path)
    scenario_name = cfg["experiment"]["name"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for i in range(n_runs):
        print(f"\n{'='*50}")
        print(f"  Scenario: {scenario_name}  |  Run {i + 1}/{n_runs}")
        print(f"{'='*50}")
        # deepcopy so each run gets fresh agent state and a clean logger
        data = await _run_single(deepcopy(cfg), steps)
        df = pd.DataFrame(data)
        df["run"] = i
        all_dfs.append(df)
        df.to_csv(out / f"{scenario_name}_run_{i}.csv", index=False)
        print(f"  Run {i + 1} saved.")

    combined = pd.concat(all_dfs, ignore_index=True)
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    grouped = combined.groupby("step")[numeric_cols]

    avg_df = grouped.mean()
    std_df = grouped.std()

    avg_df.to_csv(out / f"{scenario_name}_avg.csv")
    std_df.to_csv(out / f"{scenario_name}_std.csv")

    print(f"\nResults for '{scenario_name}' saved to {out}/")
    print(f"  {scenario_name}_avg.csv  ({len(avg_df)} steps, {len(avg_df.columns)} columns)")
    print(f"  {scenario_name}_std.csv")


def main():
    parser = argparse.ArgumentParser(description="Run repeated simulations for DSGE IRF validation")
    parser.add_argument("--config", default="config/experiments/baseline.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of simulation runs to average over")
    parser.add_argument("--steps", type=int, default=30,
                        help="Number of steps per simulation run")
    parser.add_argument("--out", default="data/experiments",
                        help="Output directory for averaged CSVs")
    args = parser.parse_args()

    asyncio.run(run_experiment(args.config, args.runs, args.steps, args.out))


if __name__ == "__main__":
    main()
