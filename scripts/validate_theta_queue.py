"""Validation: re-evaluate top-K theta candidates under queue clearing.

Loads data/msm/lhs_results.jsonl, picks top-K candidates by J under live
FRED, re-runs each ONCE with goods_clearing_mode='queue', and saves results
to data/msm/queue_validation.jsonl.

If the ranking under queue matches the ranking under average within top-K,
calibration is robust to goods market specification.

Usage:
    python scripts/validate_theta_queue.py --top 3 --steps 50 --households 20
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

from dotenv import load_dotenv
load_dotenv()

from src.analysis.fred_targets import get_target_moments
from src.analysis.moments import moments_from_sim_csv
from src.analysis.msm import J, deep_merge, theta_to_config_patch
from src.core.engine import SimulationEngine
from src.llm.factory import get_llm_backend
from src.utils.config import load_config


def override_hh(cfg: dict, n: int) -> dict:
    cfg = deepcopy(cfg)
    for g in cfg.get("agents", {}).values():
        if g.get("type") == "household":
            g["count"] = n
            break
    return cfg


async def _run(cfg: dict, steps: int) -> pd.DataFrame:
    llm = get_llm_backend(cfg)
    engine = SimulationEngine(llm, cfg)
    await engine.run(steps=steps)
    return pd.DataFrame(engine.logger.data)


async def main_async(args):
    msm_path = Path("data/msm/lhs_results.jsonl")
    out_path = Path("data/msm/queue_validation.jsonl")

    with open(msm_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    target = get_target_moments()
    rescored = []
    for r in records:
        j, _ = J(r["sim_moments"], target)
        rescored.append({**r, "J_avg": j})

    sorted_r = sorted(rescored, key=lambda r: r["J_avg"])
    top = sorted_r[: args.top]
    print(f"Top-{args.top} θ under average mode:")
    for i, r in enumerate(top):
        print(f"  {i+1}. {r['label']}: J_avg={r['J_avg']:.2f}, A={r['theta']['match_eff']:.2f}, price_adj={r['theta']['price_adj']:.2f}")

    # Resume support
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["label"])

    base_cfg = load_config(args.base)
    # Force queue mode
    base_cfg.setdefault("market", {})["goods_clearing_mode"] = "queue"

    for r in top:
        if r["label"] in done:
            print(f"  [{r['label']}] already validated, skipping")
            continue
        print(f"\n=== Validate {r['label']} under queue ===")
        cfg = deep_merge(base_cfg, theta_to_config_patch(r["theta"]))
        cfg = override_hh(cfg, args.households)
        cfg["experiment"] = {**cfg.get("experiment", {}), "name": f"qvalid_{r['label']}"}
        try:
            df = await _run(cfg, args.steps)
            sim_m_q = moments_from_sim_csv(df)
            j_q, parts_q = J(sim_m_q, target)
            rec = {
                "label": r["label"],
                "theta": r["theta"],
                "J_avg_baseline": r["J_avg"],
                "J_queue": j_q,
                "sim_moments_queue": sim_m_q,
            }
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            print(f"  J under queue = {j_q:.2f}  (was {r['J_avg']:.2f} under average)")
            print(f"  unrate under queue = {sim_m_q.get('unrate_mean', 0):.3f}")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Report
    print("\n=== Queue validation summary ===")
    with open(out_path) as f:
        results = [json.loads(l) for l in f if l.strip()]
    results.sort(key=lambda r: r["J_queue"])
    print(f"{'rank':<6}{'label':<10}{'J_avg':<10}{'J_queue':<10}{'Δ':<10}")
    for i, r in enumerate(results):
        delta = r["J_queue"] - r["J_avg_baseline"]
        print(f"{i+1:<6}{r['label']:<10}{r['J_avg_baseline']:<10.2f}{r['J_queue']:<10.2f}{delta:+10.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--households", type=int, default=20)
    p.add_argument("--base", default="config/experiments/msm_base.yaml")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
