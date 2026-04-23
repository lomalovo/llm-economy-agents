"""Counterfactual-bio driver (resume-aware).

Runs the four bio-composition conditions (all-HtM, all-saver, mixed, 50/50),
N_RUNS times each, writing per-run CSV to `--out`. On restart, skips any
condition/run that already has a CSV, so crashes cost nothing.

Usage:
    python scripts/run_counterfactual.py --runs 2 --steps 40
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config


CONDITIONS = ["cf_all_htm", "cf_all_saver", "cf_mixed", "cf_fifty_fifty"]


async def _run_once(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


async def main_async(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"

    for cond in CONDITIONS:
        existing = list(out_dir.glob(f"{cond}_run_*.csv"))
        if len(existing) >= args.runs:
            print(f"{cond}: already has {len(existing)} CSVs, skipping")
            continue
        # Determine which run ids are missing
        have_ids = set()
        for p in existing:
            # format: cond_run_N.csv
            try:
                rid = int(p.stem.split("_")[-1])
                have_ids.add(rid)
            except Exception:
                pass
        missing = [i for i in range(args.runs) if i not in have_ids]

        cfg_path = f"config/experiments/{cond}.yaml"
        print(f"\n===== {cond}: running missing ids={missing} =====", flush=True)
        for i in missing:
            run_id = f"{cond}_run_{i}"
            t0 = time.time()
            cfg = load_config(cfg_path)
            try:
                df = await _run_once(deepcopy(cfg), args.steps)
                out_csv = out_dir / f"{run_id}.csv"
                df.to_csv(out_csv, index=False)
                elapsed = time.time() - t0
                with open(progress_path, "a") as f:
                    f.write(json.dumps({
                        "condition": cond, "run": i, "status": "ok",
                        "steps": args.steps, "elapsed_s": round(elapsed, 1),
                        "final_unrate": float(df["unemployment_rate"].iloc[-1]),
                        "final_output": float(df["total_sales"].iloc[-1]),
                    }) + "\n")
                print(f"  {run_id}: done in {elapsed:.0f}s ({out_csv})", flush=True)
            except Exception as e:
                elapsed = time.time() - t0
                with open(progress_path, "a") as f:
                    f.write(json.dumps({
                        "condition": cond, "run": i, "status": "failed",
                        "error": str(e), "elapsed_s": round(elapsed, 1),
                    }) + "\n")
                print(f"  {run_id}: FAILED after {elapsed:.0f}s — {e}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--out", default="data/counterfactual")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
