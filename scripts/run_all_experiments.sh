#!/bin/bash
# End-to-end pipeline orchestrator.
# Each phase is skipped if its output artefact already exists.
# Kick-off individual phases via env: SKIP_CF=1 SKIP_MSM=1 etc.

set -euo pipefail

source venv/bin/activate
mkdir -p data/counterfactual data/msm data/audit data/scale_ablation charts/v2

# ---- Phase 4: scale ablation (dummy, free) ----
if [[ ! -f data/scale_ablation/scale_summary.csv && -z "${SKIP_SCALE:-}" ]]; then
  echo
  echo "=== Phase 4: scale ablation ==="
  python scripts/run_scale_ablation.py 2>&1 | tee data/scale_ablation.log
fi

# ---- Phase 2: counterfactual (LLM) ----
if [[ ! -f data/counterfactual/progress.jsonl && -z "${SKIP_CF:-}" ]]; then
  echo
  echo "=== Phase 2: counterfactual bio ==="
  python scripts/run_counterfactual.py --runs 2 --steps 40 --out data/counterfactual 2>&1 | tee data/counterfactual/driver.log
fi

# ---- Phase 3: MSM calibration (LLM) ----
if [[ ! -f data/msm/state.json && -z "${SKIP_MSM:-}" ]]; then
  echo
  echo "=== Phase 3: MSM LHS grid ==="
  python run_msm.py \
    --base config/experiments/msm_base.yaml \
    --lhs 15 --steps 60 --households 30 \
    --out data/msm 2>&1 | tee data/msm/lhs.log

  echo
  echo "=== Phase 3: MSM refinement ==="
  python run_msm.py \
    --refine data/msm/state.json \
    --refine-runs 3 --steps 80 --households 50 \
    --out data/msm 2>&1 | tee data/msm/refine.log
fi

# ---- Phase 5: final hero IRF at θ* ----
if [[ ! -f data/hero/baseline_run_0.csv && -z "${SKIP_HERO:-}" ]]; then
  echo
  echo "=== Phase 5: hero IRF runs ==="
  python scripts/run_hero_irf.py --theta data/msm/state.json --runs 4 --steps 80 --households 50
fi

# ---- Phase 6: narrative audit ----
if [[ ! -f data/audit/summary.json && -z "${SKIP_AUDIT:-}" ]]; then
  echo
  echo "=== Phase 6: narrative audit ==="
  # Concatenate all counterfactual + hero reasoning JSONLs and audit them.
  python scripts/run_narrative_audit.py
fi

# ---- Analysis & charts ----
echo
echo "=== Analysis: counterfactual summary ==="
python scripts/analyze_counterfactual.py || true

echo
echo "=== Charts ==="
python scripts/make_paper_charts.py

echo
echo "Pipeline complete. Artefacts:"
ls -1 data/msm/*.json data/msm/*.csv data/counterfactual/analysis/*.json charts/v2/*.png 2>/dev/null
