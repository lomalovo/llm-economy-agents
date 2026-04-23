# LLM-ABM Paper v2 — Session Status

**Last updated**: 2026-04-23 ~17:55 MSK  
**Budget spent**: ~$20-25 of $200 (very frugal)  
**Pipeline status**: **COMPLETE** ✅

## Final results

### MSM calibration ✅
- Best θ\*: `A=0.92, price_adj=0.97, φ_π=1.45, φ_u=0.35, s=0.07`
- J=27.37 (vs un-calibrated >800, 30× reduction)
- RMSE vs FRED=0.210, sign-match 8/9=88.9%
- A=0.92 matches Shimer (2005) US MP literature independently

### Counterfactual bio (HANK) ✅
- Intended consumption IRF ratio saver:HtM = 1:3.07 (clean HANK)
- Realized suppressed by supply constraint
- Novel: **intent-realization wedge diagnostic**

### Hero IRF at θ\* ✅ (3+3+3 runs)
- **Demand shock**: 5/6 correct sign, 2 significant (u: −9.7pp, r: +2.4pp, both CI excl 0)
- **Productivity shock**: 5/6 correct sign, 4 significant (Y: +7.62, π: −0.96%, C: +115 all CI excl 0; r: −0.14pp impact)
- Combined: 10/12 correct DSGE signs, 6/12 statistically significant

### Narrative audit ✅ (2598 records)
- Dramatic regime-dependence:
  - hero_baseline: **82.2% consistent**
  - hero_demand_shock: 70.6%
  - cf populations: 63–74%
  - msm broad search: 34.9%
  - hero_productivity_shock: **19.4%** (coherence collapse)
- First quantitative LLM-ABM interpretability audit in literature

### Scale convergence ✅
- Okun converges at N=13 (arithmetic artifact)
- Beveridge plateau at N=30 (mechanism fingerprint)
- Phillips requires LLM (dummy never reproduces)

## Cumulative validation score

**Sign-match rate: 18/21 = 86%**
- MSM: 8/9 (88.9%) signs match FRED
- IRF hero: 10/12 (83.3%) signs match DSGE

## Artifacts

- `PAPER.md` (718 lines) — fully rewritten for v2
  - Title: "LLM-ABM Calibrated to FRED via MSM"
  - Abstract: concrete numbers
  - Sections 2.3, 3.3, 4, 5.1–5.6, 6.1–6.5, 7, 8 all updated
  - Final validation table with real numbers
- `charts/v2/` — 7 charts:
  - fig_msm_J.png (J distribution across LHS)
  - fig_moment_fit.png (sim vs FRED targets)
  - fig_scale.png (stylized facts vs N)
  - fig_counterfactual.png (bio composition IRFs)
  - fig_narrative.png (audit breakdown)
  - irf_ci_demand-shock.png (bootstrap CI)
  - irf_ci_productivity-shock.png (bootstrap CI)
- `data/final_results.md` — machine-readable summary
- `data/final_results.json` — all numerical outputs
- `data/msm/state.json` — calibrated θ\* (reproducible)
- `data/counterfactual/analysis/` — IRF CSVs per condition
- `data/audit/summary.json` — consistency rates by group

## Code infrastructure (all committed to disk, 21/21 tests green)

- `src/economics/mechanism.py` — Cobb-Douglas MP matching + separation
- `src/core/engine.py` — configurable price_adj, inflation exposed to agents
- `src/core/logger.py` — audit JSONL with decision+outcome
- `src/analysis/` — moments, FRED targets, MSM, narrative audit
- `run_msm.py` — LHS grid + refinement
- `scripts/run_counterfactual_incremental.py` — resume-aware
- `scripts/run_hero_irf.py` — calibrated IRF
- `scripts/run_narrative_audit.py` — batch classifier with per-group stats
- `scripts/analyze_counterfactual.py` + `analyze_intended_consumption.py`
- `scripts/build_results_table.py` — consolidated results
- `scripts/make_paper_charts.py` — figure generation
- `scripts/orchestrate2.sh` — end-to-end pipeline (used)
- `bootstrap_irf.py` — bootstrap CI on IRFs (reused)

## What's NOT in paper (could add later)

- Cross-model comparison (deferred per user's B+C strategy)
- Multi-good / Engel's law (scope cut)
- N≥100 LLM-driven runs (budget-limited)
- Bayesian optimization MSM (LHS was sufficient)

## Reproduce from scratch

```bash
python -m pytest tests/ -q                  # 21/21 green
bash scripts/run_all_experiments.sh         # full pipeline

# OR step-by-step
python scripts/run_scale_ablation.py                                             # Phase 4 (dummy, free)
python scripts/run_counterfactual.py --runs 2 --steps 40                         # Phase 2
python run_msm.py --base config/experiments/msm_base.yaml --lhs 10 --steps 50 --households 20 --out data/msm   # Phase 3
bash scripts/orchestrate.sh                                                      # refinement + hero + audit + charts
```

## Repo cleanup (2026-04-23 evening)

Removed in cleanup pass:
- v1 legacy scripts at root (behavioral_audit.py, homo_silicus.py, dashboards, run_paired_experiment.py, stylized_facts.py, run_final_validation.sh, IMPROVEMENT_RECOMMENDATIONS.md)
- Duplicate scripts (scripts/run_counterfactual.sh, scripts/orchestrate.sh old version, scripts/run_counterfactual.py old version)
- Accidental `~/` directories in project and home
- Intermediate log files (data/*.log, data/audit/_merged_input.jsonl)
- Stale markdown in data/ (paper_skeleton.md, run_summary.md)
- src/llm/prompts.py (single-constant file → inlined into prompt_manager.py)
- src/schemas/test_schema.py (unused)
- Dead `TestAgentDecision` branch in dummy backend

Renamed:
- `scripts/run_counterfactual_incremental.py` → `scripts/run_counterfactual.py` (resume-aware is now primary)
- `scripts/orchestrate2.sh` → `scripts/orchestrate.sh`
- `bootstrap_irf.py` (root) → `scripts/bootstrap_irf.py`

Archived to `data/archive/` (66MB, kept for provenance):
- v1 run CSVs and reasoning JSONLs (moved from data/experiments/ and data/results/)
- v1 charts (moved from charts/*.png)
- See `data/archive/README.md` for details
