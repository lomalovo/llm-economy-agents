# LLM-ABM Calibrated to FRED via MSM

Agent-based macroeconomic simulation where households and firms replace
hand-coded behavioural rules with per-step calls to an LLM. Each tick, 10–50
LLM agents decide on labour supply, consumption, production, and prices based
on their biographies, history, and the current macro state; the market
mechanism (Mortensen–Pissarides matching + Walrasian goods clearing + Taylor
rule + progressive tax) clears synchronously.

The current paper (see `PAPER.md`) calibrates this system to US FRED quarterly
moments via Method of Simulated Moments, tests HANK-style predictions via
counterfactual bio composition, audits reasoning↔action consistency, and
establishes scale-convergence properties.

**Headline numbers (v2, April 2026)**:
- **MSM**: best J=27.37, RMSE=0.21 vs FRED, **sign-match 8/9 = 89%**.
  Best θ\*=(A=0.92, φ_π=1.45, price_adj=0.97, φ_u=0.35, s=0.07) —
  MSM independently rediscovers US MP literature estimates.
- **Hero IRF**: **10/12 DSGE-correct signs** across demand + productivity
  shocks; 6/12 statistically significant at shock step.
- **Counterfactual bio**: intended-consumption IRF ratio saver:HtM = 1:3.07 —
  clean HANK pattern. Supply-constrained realized IRF → novel
  **intent–realization wedge** diagnostic.
- **Narrative audit**: reasoning↔action consistency ranges from **82%
  (stable baseline)** to **19% (productivity shock)** — first quantitative
  LLM-ABM interpretability audit.

## Setup

Python 3.10+ and an API key for an OpenAI-compatible LLM service.

```bash
git clone https://github.com/lomalovo/llm-economy-agents.git
cd llm-economy-agents
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env`:

```
ELIZA_API_KEY=y1_...           # preferred: OpenRouter proxy via Yandex Eliza
DEEPSEEK_API_KEY=sk-...        # fallback: direct DeepSeek API
FRED_API_KEY=...               # optional: for live FRED target moments (falls back to hardcoded if absent)
```

Backend selected in config via `llm.backend_type`:
- `eliza` — Eliza/OpenRouter proxy → DeepSeek-V3 (default for v2 experiments)
- `deepseek` — direct DeepSeek API
- `openai` — OpenAI or compatible
- `dummy` — no API calls, random valid decisions (use for logic testing)

## Quick run

```bash
# Dummy smoke test (free)
python main.py   # uses config/config.yaml

# Paper pipeline (full, ~4 hours, ~$25 API spend on DeepSeek via Eliza)
bash scripts/run_all_experiments.sh

# Or step-by-step
python scripts/run_counterfactual.py --runs 2 --steps 40            # Phase 2
python run_msm.py --base config/experiments/msm_base.yaml --lhs 10 --steps 50 --households 20  # Phase 3 part 1
bash scripts/orchestrate.sh > data/orchestrate.log 2>&1 &           # remaining phases
```

## Architecture at a glance

```
config YAML                                 data/results/*.csv + *_reasoning.jsonl
    │                                                    ▲
    ▼                                                    │
AgentBuilder → households / firms (LLM)  ──► SimulationEngine.step()
                                              ├── _process_events()       (cash inject / TFP shock)
                                              ├── parallel LLM decisions  (make_decision per agent)
                                              ├── clear_labor_market      (MP matching, sep rate)
                                              ├── _collect_taxes_and_redistribute
                                              ├── clear_goods_market      (average / queue / weighted)
                                              ├── price update            (Calvo partial adjustment)
                                              ├── _update_interest_rate   (Taylor rule)
                                              ├── _apply_interest_on_savings
                                              └── log_step → CSV + JSONL
```

Step-lifecycle details: see `src/core/engine.py::SimulationEngine.step()`. Per-agent context: `src/agents/impl.py`.

## Directory map

| Path | Purpose |
|------|---------|
| `PAPER.md` | the paper (v2) with final numbers and charts referenced |
| `STATUS.md` | live session log with reproduce instructions |
| `config/experiments/` | per-scenario YAMLs |
| `src/core/`, `src/agents/`, `src/economics/`, `src/llm/`, `src/analysis/` | core code |
| `scripts/` | pipeline + analysis drivers |
| `tests/` | pytest suite (21 tests) |
| `templates/` | Jinja2 agent prompts |
| `data/counterfactual/`, `data/msm/`, `data/hero/`, `data/audit/`, `data/scale_ablation/` | experimental outputs |
| `data/archive/` | v1 artefacts kept for provenance |
| `charts/v2/` | final paper figures |

## Tests

```bash
python -m pytest tests/ -q   # 21/21 green
```

## Citation

If you use this work, please cite `PAPER.md` — "Minds in the Market: LLM-ABM
Calibrated to FRED via MSM" (Nazarov, 2026).
