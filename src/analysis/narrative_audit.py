"""Narrative causal audit.

For each (reasoning, decision, outcome) triple in an audit JSONL, ask an LLM
classifier whether the reasoning is consistent with the numerical decision.

Why this matters: an LLM-ABM is only interpretable if the generated reasoning
*actually* caused the action. If agents produce plausible-sounding text but
their numerical decisions are uncorrelated with it, the reasoning is
confabulation and the interpretability claim fails. The consistency rate is
the first diagnostic for this.

Labels:
    consistent    — reasoning logically entails the numerical decision
    inconsistent  — reasoning contradicts or ignores the decision
    unclear       — reasoning too vague / contains no causal content

Usage:
    python -m src.analysis.narrative_audit --audit data/results/foo_reasoning.jsonl \
        --out data/audit/foo_audit.jsonl --batch-size 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm.factory import get_llm_backend


class ConsistencyJudgment(BaseModel):
    label: str = Field(..., description="One of: consistent, inconsistent, unclear")
    explanation: str = Field(..., description="One short sentence justifying the label")


JUDGE_SYSTEM = (
    "You are an auditor checking whether an economic agent's free-text reasoning "
    "is consistent with the numerical decision that agent then produced. "
    "Respond with a short judgment in JSON."
)


def _build_prompt(record: dict) -> str:
    agent_type = record.get("agent_type", "household")
    decision = record.get("decision", {})
    outcome = record.get("outcome", {})
    reasoning = record.get("reasoning", "").strip()
    dec_str = ", ".join(f"{k}={v}" for k, v in decision.items())
    out_str = ", ".join(f"{k}={v}" for k, v in outcome.items())
    return (
        f"Agent type: {agent_type}\n"
        f"Agent reasoning (free text):\n\"{reasoning}\"\n\n"
        f"Numerical decision: {dec_str}\n"
        f"Realized outcome: {out_str}\n\n"
        f"Label this pair:\n"
        f"- consistent:   the reasoning logically leads to (or is compatible with) the numerical decision.\n"
        f"- inconsistent: the reasoning states intentions that contradict the decision "
        f"(e.g., \"I will cut spending\" but consumption_budget is very high).\n"
        f"- unclear:      the reasoning is too vague, generic, or missing causal content "
        f"to judge the link to the decision.\n"
        f"Reply with JSON: {{\"label\": \"...\", \"explanation\": \"...\"}}"
    )


async def _judge_one(llm, record: dict) -> dict:
    prompt = _build_prompt(record)
    try:
        result = await llm.generate(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=prompt,
            schema=ConsistencyJudgment,
        )
        return {
            "step":     record.get("step"),
            "agent_id": record.get("agent_id"),
            "label":    result.label,
            "explanation": result.explanation,
        }
    except Exception as e:
        return {
            "step":     record.get("step"),
            "agent_id": record.get("agent_id"),
            "label":    "error",
            "explanation": f"classifier error: {e}",
        }


async def audit_jsonl(
    audit_path: Path,
    out_path: Path,
    backend_type: str = "deepseek",
    batch_size: int = 20,
    max_records: int | None = None,
    resume: bool = True,
) -> dict:
    """Run narrative consistency classification over an audit JSONL."""
    # Build a throwaway config for the classifier backend
    model_name = {
        "deepseek": "deepseek-chat",
        "eliza":    "deepseek/deepseek-chat-v3-0324",
        "openai":   "gpt-4o-mini",
    }.get(backend_type, None)
    cfg = {
        "llm": {
            "backend_type": backend_type,
            "model_name": model_name,
            "max_concurrency": batch_size,
            "max_retries": 3,
            "timeout": 60,
        }
    }
    llm = get_llm_backend(cfg)

    records = []
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if max_records:
        records = records[:max_records]

    done_keys = set()
    if resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r.get("step"), r.get("agent_id")))
                except Exception:
                    pass

    to_run = [r for r in records if (r.get("step"), r.get("agent_id")) not in done_keys]
    print(f"[narrative-audit] {len(to_run)} records to judge (skipping {len(records) - len(to_run)} existing)")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def process_batch(batch):
        tasks = [_judge_one(llm, r) for r in batch]
        results = await asyncio.gather(*tasks)
        with open(out_path, "a") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return results

    # Process in batches to keep concurrency bounded
    stats = {"consistent": 0, "inconsistent": 0, "unclear": 0, "error": 0}
    for i in range(0, len(to_run), batch_size):
        batch = to_run[i : i + batch_size]
        results = await process_batch(batch)
        for r in results:
            stats[r.get("label", "error")] = stats.get(r.get("label", "error"), 0) + 1
        total_so_far = sum(stats.values())
        print(f"  {i + len(batch)}/{len(to_run)} judged | running totals: {stats}")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit", required=True, help="path to *_reasoning.jsonl from a run")
    p.add_argument("--out", required=True, help="path to write judgments jsonl")
    p.add_argument("--backend", default="deepseek")
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--max", type=int, default=None, help="cap total records (for testing)")
    args = p.parse_args()

    stats = asyncio.run(audit_jsonl(
        audit_path=Path(args.audit),
        out_path=Path(args.out),
        backend_type=args.backend,
        batch_size=args.batch_size,
        max_records=args.max,
    ))
    print(f"\nFinal stats: {stats}")
    total = sum(stats.values())
    if total:
        for k, v in stats.items():
            print(f"  {k}: {v / total:.1%}")


if __name__ == "__main__":
    main()
