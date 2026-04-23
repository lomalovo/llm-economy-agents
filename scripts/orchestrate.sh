#!/bin/bash
# Post-LHS orchestrator. Launch AFTER `run_msm.py` is running (or done) —
# waits for data/msm/state.json to exist, then runs refinement → hero IRF →
# narrative audit → final charts + results table.
#
# Expected wall-clock: ~2-4h depending on API latency and N/steps settings.

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

STATE=data/msm/state.json

echo "[orchestrate] Waiting for $STATE (MSM LHS completion)..."
while [[ ! -f "$STATE" ]]; do
  if ! pgrep -f "run_msm.py" > /dev/null; then
    echo "[orchestrate] MSM main process gone but $STATE missing — aborting"
    exit 1
  fi
  sleep 60
done
echo "[orchestrate] $STATE exists at $(date)"

# Also wait for cf to fully finish so hero IRF starts on a clean API
echo "[orchestrate] Waiting for counterfactual driver to stop..."
while pgrep -f "run_counterfactual.py" > /dev/null; do
  sleep 30
done

echo
echo "[orchestrate] === Phase 2 final analysis ==="
python scripts/analyze_counterfactual.py || true
python scripts/analyze_intended_consumption.py \
  --conditions cf_all_htm cf_all_saver cf_mixed cf_fifty_fifty || true

echo
echo "[orchestrate] === Phase 3: MSM refinement ==="
python run_msm.py \
  --refine data/msm/state.json \
  --refine-runs 3 --steps 80 --households 30 \
  --out data/msm

echo
echo "[orchestrate] === Phase 5: Hero IRF ==="
python scripts/run_hero_irf.py \
  --theta data/msm/state.json \
  --runs 3 --steps 60 --households 30 \
  --out data/hero

echo
echo "[orchestrate] === Phase 6: Narrative audit ==="
python scripts/run_narrative_audit.py \
  --sources data/results \
  --prefixes cf_all_htm cf_all_saver cf_mixed cf_fifty_fifty hero_ msm_ \
  --max-per-source 60 --max-total 1200 --batch-size 10 \
  --backend eliza \
  --out data/audit/judgments.jsonl

echo
echo "[orchestrate] === Phase 7: build results + charts ==="
python scripts/build_results_table.py
python scripts/make_paper_charts.py

echo
echo "[orchestrate] DONE at $(date)"
ls data/final_results.md data/final_results.json charts/v2/*.png 2>&1 || true
