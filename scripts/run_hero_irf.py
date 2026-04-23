"""Phase 5 — final hero IRF runs at calibrated θ*.

Loads MSM state.json, constructs a calibrated base config, then runs
baseline / demand_shock / productivity_shock each N times with bootstrap
CI over IRFs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config
from src.analysis.msm import deep_merge, theta_to_config_patch


SCENARIOS = {
    "baseline": "config/experiments/baseline.yaml",
    "demand_shock": "config/experiments/demand_shock.yaml",
    "productivity_shock": "config/experiments/productivity_shock.yaml",
}


def override_household_count(cfg: dict, n: int) -> dict:
    cfg = deepcopy(cfg)
    for group in cfg.get("agents", {}).values():
        if group.get("type") == "household":
            group["count"] = n
            break
    return cfg


async def _run_once(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


async def main_async(args):
    with open(args.theta) as f:
        state = json.load(f)
    theta = state["best"]["theta"]
    patch = theta_to_config_patch(theta)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / "progress.jsonl"

    for scenario, cfg_path in SCENARIOS.items():
        print(f"\n=== {scenario} x {args.runs} runs (θ*) ===")
        for i in range(args.runs):
            cfg = load_config(cfg_path)
            cfg = deep_merge(cfg, patch)
            if args.households:
                cfg = override_household_count(cfg, args.households)
            cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"hero_{scenario}_{i}"}

            try:
                df = await _run_once(deepcopy(cfg), args.steps)
                out_csv = out_dir / f"{scenario}_run_{i}.csv"
                df.to_csv(out_csv, index=False)
                with open(progress_path, "a") as f:
                    f.write(json.dumps({
                        "scenario": scenario, "run": i, "status": "ok",
                        "final_unrate": float(df["unemployment_rate"].iloc[-1]),
                    }) + "\n")
                print(f"  {scenario}_run_{i}: saved to {out_csv}")
            except Exception as e:
                with open(progress_path, "a") as f:
                    f.write(json.dumps({
                        "scenario": scenario, "run": i, "status": "failed",
                        "error": str(e),
                    }) + "\n")
                print(f"  {scenario}_run_{i}: FAILED — {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--theta", default="data/msm/state.json")
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--households", type=int, default=50)
    p.add_argument("--out", default="data/hero")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
