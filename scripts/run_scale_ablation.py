"""Scale convergence ablation (Phase 4).

Runs the same baseline config at N=13, 30, 50, 100, 200 households with dummy
backend, 60 steps each, 3 runs per N. Computes stylized facts at each N and
plots |fact - benchmark| vs. N — answers the methodological question:
"how many agents does each stylized fact need before it becomes detectable?"

Dummy backend is used here deliberately: we want the convergence curve for the
MATCHING MECHANISM at large N, not the LLM narrative layer. The LLM runs at
N=30 (counterfactual) and N=50 (MSM, hero) separately.
"""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.analysis.moments import moments_from_sim_csv


async def _run_single(cfg: dict, steps: int) -> list[dict]:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return engine.logger.data


N_VALUES = [13, 30, 50, 100, 200]
N_RUNS = 3
STEPS = 60


def make_base_cfg(n_hh: int) -> dict:
    return {
        "experiment": {
            "name": f"scale_N{n_hh}",
            "description": f"Scale ablation N={n_hh}",
        },
        "initial_state": {"avg_price": 15.0, "avg_wage": 15.0, "prev_avg_price": 15.0},
        "market": {
            "goods_clearing_mode": "average",
            "wage_adjustment_speed": 0.15,
            "matching_efficiency": 0.6,
            "matching_elasticity": 0.5,
            "separation_rate": 0.05,
            "price_adjustment_speed": 0.4,
        },
        "simulation": {"history_window": 5, "reflection_every": 10000},
        "central_bank": {
            "enabled": True, "neutral_rate": 0.05, "target_inflation": 0.02,
            "target_unemployment": 0.05, "inflation_sensitivity": 1.5,
            "unemployment_sensitivity": 0.5, "min_rate": 0.0, "max_rate": 0.25,
        },
        "government": {
            "enabled": True,
            "tax_brackets": [
                {"threshold": 0, "rate": 0.10},
                {"threshold": 20, "rate": 0.20},
                {"threshold": 50, "rate": 0.35},
            ],
        },
        "llm": {"backend_type": "dummy"},
        "events": [],
        "agents": {
            "households": {
                "type": "household",
                "count": n_hh,
                "params": {
                    "initial_money": 200.0,
                    "template": "household.j2",
                    "bio": "Generic household under dummy backend.",
                },
            },
            "firms": {
                "type": "firm",
                "count": max(2, n_hh // 5),  # scale firms proportionally
                "params": {
                    "initial_capital": 1000.0,
                    "template": "firm.j2",
                    "bio": "Generic firm under dummy backend.",
                },
            },
        },
    }


async def run_for_n(n_hh: int, n_runs: int, steps: int) -> list[dict]:
    results = []
    for i in range(n_runs):
        print(f"  N={n_hh} run {i + 1}/{n_runs}...", flush=True)
        df = pd.DataFrame(await _run_single(deepcopy(make_base_cfg(n_hh)), steps))
        moments = moments_from_sim_csv(df)
        results.append({"n": n_hh, "run": i, "moments": moments})
    return results


async def main():
    out_dir = Path("data/scale_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for n in N_VALUES:
        recs = await run_for_n(n, N_RUNS, STEPS)
        all_results.extend(recs)
        # checkpoint after each N
        with open(out_dir / "raw_results.jsonl", "a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")

    # Aggregate
    rows = []
    for r in all_results:
        row = {"n": r["n"], "run": r["run"], **r["moments"]}
        rows.append(row)
    df = pd.DataFrame(rows)

    # Per-N means
    agg = df.groupby("n").mean(numeric_only=True).reset_index()
    agg.to_csv(out_dir / "scale_summary.csv", index=False)
    print(f"\nSaved aggregate to {out_dir / 'scale_summary.csv'}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
