#!/bin/bash
# Triggered after perturb_around_best.py finishes:
#   1. Re-score all θ (incl. new perturbations) under live FRED + IRF quality
#   2. Pick top-2 winners
#   3. Hero IRF runs at each winner (5 each scenario, weighted + history=10)
#   4. Comparison vs DSGE/RBC + final charts

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "[fin_pert] Waiting for perturbation to finish..."
while pgrep -f "perturb_around_best.py" > /dev/null; do
  sleep 60
done
echo "[fin_pert] perturbation done at $(date)"

echo
echo "[fin_pert] === Step 1: multi-criteria ranking ==="
python scripts/score_theta_irf.py --top-n 15 --alpha 0.5

# Pull top-2 winners from ranking
TOP1_LABEL=$(python3 -c "
import json
with open('data/msm/theta_ranking.json') as f: r = json.load(f)
print(r[0]['label'])
")
TOP2_LABEL=$(python3 -c "
import json
with open('data/msm/theta_ranking.json') as f: r = json.load(f)
print(r[1]['label'])
")
echo "[fin_pert] Top-1 = $TOP1_LABEL, Top-2 = $TOP2_LABEL"

# Build state.json for each winner via msm_full_analysis (with override)
echo
echo "[fin_pert] === Step 2a: Hero IRF for top-1 winner ($TOP1_LABEL) ==="
python3 << EOF
import json
with open('data/msm/lhs_results.jsonl') as f:
    records = [json.loads(l) for l in f if l.strip()]
win = next(r for r in records if r['label'] == '$TOP1_LABEL')
state = {'best': win, 'target': None, 'target_source': 'live FRED', 'note': 'top-1 by combined J+IRFq score'}
from src.analysis.fred_targets import get_target_moments
state['target'] = get_target_moments()
with open('data/msm/state_winner1.json','w') as f:
    json.dump(state, f, indent=2, default=float)
print(f"Saved state_winner1.json: $TOP1_LABEL")
EOF

python scripts/run_hero_irf.py \
  --theta data/msm/state_winner1.json \
  --runs 5 --steps 60 --households 30 \
  --goods-mode weighted \
  --out data/hero_weighted_w1

echo
echo "[fin_pert] === Step 2b: Hero IRF for top-2 winner ($TOP2_LABEL) ==="
python3 << EOF
import json
with open('data/msm/lhs_results.jsonl') as f:
    records = [json.loads(l) for l in f if l.strip()]
win = next(r for r in records if r['label'] == '$TOP2_LABEL')
state = {'best': win, 'target': None, 'target_source': 'live FRED', 'note': 'top-2 by combined J+IRFq score'}
from src.analysis.fred_targets import get_target_moments
state['target'] = get_target_moments()
with open('data/msm/state_winner2.json','w') as f:
    json.dump(state, f, indent=2, default=float)
EOF

python scripts/run_hero_irf.py \
  --theta data/msm/state_winner2.json \
  --runs 5 --steps 60 --households 30 \
  --goods-mode weighted \
  --out data/hero_weighted_w2

echo
echo "[fin_pert] === Step 3: IRF comparison vs DSGE/RBC ==="
HERO_DATA_DIR=data/hero_weighted_w1 python scripts/compare_irf_dsge.py
# Rename winner-1 outputs
mv -f charts/v2/fig_irf_vs_dsge_demand.png charts/v2/fig_irf_vs_dsge_demand_w1.png 2>/dev/null
mv -f charts/v2/fig_irf_vs_rbc_productivity.png charts/v2/fig_irf_vs_rbc_productivity_w1.png 2>/dev/null

HERO_DATA_DIR=data/hero_weighted_w2 python scripts/compare_irf_dsge.py
mv -f charts/v2/fig_irf_vs_dsge_demand.png charts/v2/fig_irf_vs_dsge_demand_w2.png 2>/dev/null
mv -f charts/v2/fig_irf_vs_rbc_productivity.png charts/v2/fig_irf_vs_rbc_productivity_w2.png 2>/dev/null

echo
echo "[fin_pert] === Step 4: Final build ==="
python scripts/msm_full_analysis.py
python scripts/build_results_table.py
python scripts/make_paper_charts.py

echo
echo "[fin_pert] DONE at $(date)"
ls data/hero_weighted_w*/  data/final_results.md charts/v2/fig_irf_*.png 2>/dev/null
