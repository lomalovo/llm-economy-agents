"""Phase 6 — narrative audit over collected reasoning files.

Concatenates all *_reasoning.jsonl from data/results/ that correspond to our
experiments (cf_*, hero_*, msm_*), classifies each record's
reasoning-action consistency, and saves aggregated stats.

Sampling is applied (--max-per-source) to stay under budget.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.narrative_audit import audit_jsonl


def collect_records(sources: list[Path], max_per_source: int, seed: int = 42) -> Path:
    """Merge selected audit JSONLs into a single sampled file."""
    random.seed(seed)
    out_path = Path("data/audit/_merged_input.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as out:
        for src in sources:
            with open(src) as f:
                lines = f.readlines()
            if len(lines) > max_per_source:
                lines = random.sample(lines, max_per_source)
            for line in lines:
                line = line.strip()
                if line:
                    # tag source for later grouping
                    try:
                        rec = json.loads(line)
                        rec["source_file"] = src.name
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        n += 1
                    except Exception:
                        continue
    print(f"Merged {n} records from {len(sources)} sources → {out_path}")
    return out_path


def discover_sources(pattern_dirs: list[str]) -> list[Path]:
    sources = []
    for pat in pattern_dirs:
        base = Path(pat)
        if base.is_dir():
            sources.extend(sorted(base.glob("*_reasoning.jsonl")))
        else:
            sources.extend(sorted(Path(".").glob(pat)))
    return sources


async def main_async(args):
    sources = discover_sources(args.sources)
    sources = [s for s in sources if any(s.name.startswith(p) for p in args.prefixes)]
    print(f"Discovered {len(sources)} audit JSONLs:")
    for s in sources:
        print(f"  {s}")
    if not sources:
        print("No audit JSONLs found — nothing to audit.")
        return

    merged = collect_records(sources, args.max_per_source)

    out_path = Path(args.out)
    stats = await audit_jsonl(
        audit_path=merged,
        out_path=out_path,
        backend_type=args.backend,
        batch_size=args.batch_size,
        max_records=args.max_total,
    )

    # Save summary with per-source breakdown
    summary_path = out_path.parent / "summary.json"
    # Reload judgments to regroup by source
    by_source = {}
    with open(merged) as f:
        merged_records = [json.loads(l) for l in f if l.strip()]
    judgments = {}
    with open(out_path) as f:
        for line in f:
            j = json.loads(line)
            judgments[(j["step"], j["agent_id"])] = j["label"]
    for rec in merged_records:
        key = (rec["step"], rec["agent_id"])
        source = rec.get("source_file", "unknown")
        # Group by condition prefix (cf_all_htm, cf_all_saver, etc.)
        for prefix in ["cf_all_htm", "cf_all_saver", "cf_mixed", "cf_fifty_fifty", "hero_baseline", "hero_demand_shock", "hero_productivity_shock", "msm_"]:
            if source.startswith(prefix):
                group = prefix.rstrip("_") if prefix != "msm_" else "msm"
                break
        else:
            group = "other"
        label = judgments.get(key, "missing")
        by_source.setdefault(group, {"consistent": 0, "inconsistent": 0, "unclear": 0, "error": 0, "missing": 0})
        by_source[group][label] = by_source[group].get(label, 0) + 1

    total = sum(stats.values())
    rates = {k: (v / total if total else 0.0) for k, v in stats.items()}
    with open(summary_path, "w") as f:
        json.dump({
            "counts": stats, "rates": rates, "total": total,
            "by_group": by_source,
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    for k, v in stats.items():
        pct = rates[k] * 100
        print(f"  {k}: {v} ({pct:.1f}%)")
    print("\nBy group:")
    for group, counts in by_source.items():
        total_g = sum(counts.values())
        consistent_pct = counts.get("consistent", 0) / total_g * 100 if total_g else 0
        print(f"  {group:<25s} n={total_g:<4d} consistent={consistent_pct:.1f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["data/results"])
    p.add_argument("--prefixes", nargs="+", default=["cf_", "hero_", "msm_"])
    p.add_argument("--max-per-source", type=int, default=100)
    p.add_argument("--max-total", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--backend", default="eliza")
    p.add_argument("--out", default="data/audit/judgments.jsonl")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
