#!/bin/bash
# Orchestrate MSM phases AFTER initial LHS finishes:
#   B: Bayesian Optimization (5 iterations)
#   C: Refinement at new best θ* (5 runs at N=30, 80 steps)
#   D: Final analysis + chart regen
#
# Usage:
#   bash scripts/orchestrate_msm.sh
#   It waits for `pgrep run_msm.py` to return zero before continuing.

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "[orch_msm] Waiting for LHS extension to finish..."
while pgrep -f "run_msm.py" > /dev/null && ! pgrep -f "run_bo.py" > /dev/null; do
  sleep 60
done
echo "[orch_msm] LHS done at $(date)"

# Snapshot of current LHS state
LHS_COUNT=$(wc -l < data/msm/lhs_results.jsonl 2>/dev/null || echo 0)
echo "[orch_msm] LHS+BO records so far: $LHS_COUNT"

# ====== Phase B: Bayesian Optimization ======
echo
echo "[orch_msm] === Phase B: Bayesian Optimization (5 iter) ==="
python scripts/run_bo.py --n-iter 5 --steps 50 --households 20 --out data/msm

# ====== Refresh state.json with best θ* (recompute via msm_full_analysis) ======
echo
echo "[orch_msm] === Recomputing state.json with all evaluations ==="
python scripts/msm_full_analysis.py

# ====== Phase C: Refinement at new θ* ======
echo
echo "[orch_msm] === Phase C: Refinement (5 runs, N=30, 80 steps) ==="
python run_msm.py \
  --refine data/msm/state.json \
  --refine-runs 5 --steps 80 --households 30 \
  --out data/msm

# ====== Phase X: Validate top θ under queue clearing ======
echo
echo "[orch_msm] === Phase X: Queue-mode validation of top-3 θ ==="
python scripts/validate_theta_queue.py --top 3 --steps 50 --households 20

# ====== Phase H: Hero IRF runs under queue (5 per scenario, compressed) ======
echo
echo "[orch_msm] === Phase H: Hero IRF runs with queue clearing (N=5 each scenario) ==="
python scripts/run_hero_irf.py \
  --theta data/msm/state.json \
  --runs 5 --steps 60 --households 30 \
  --goods-mode queue \
  --out data/hero_queue

# ====== Phase F: Side-by-side IRF vs DSGE/RBC (using new queue hero) ======
echo
echo "[orch_msm] === Phase F: IRF comparison vs DSGE/RBC ==="
# Point compare_irf_dsge to new hero data dir
HERO_DATA_DIR=data/hero_queue python scripts/compare_irf_dsge.py

# ====== Phase G: Final build ======
echo
echo "[orch_msm] === Phase G: Build final results + charts ==="
python scripts/build_results_table.py
python scripts/make_paper_charts.py
python scripts/msm_full_analysis.py    # re-run for final A-pattern chart

echo
echo "[orch_msm] DONE at $(date)"
ls data/final_results.md data/msm/state.json data/msm/lhs_summary.csv charts/v2/*.png 2>/dev/null
