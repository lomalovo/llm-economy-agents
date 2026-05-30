# LLM-ABM Calibrated to FRED via MSM

Agent-based macroeconomic simulation where households and firms replace
hand-coded behavioural rules with per-step calls to an LLM. Each tick,
agents decide on labour supply, consumption, production, and prices based
on their biographies, history, and the current macro state. The market
mechanism — Mortensen–Pissarides matching + goods clearing + Taylor rule
+ progressive tax — clears synchronously.

The model is calibrated to US FRED quarterly moments via Method of
Simulated Moments and validated through bootstrap impulse response
functions compared against NK-DSGE and RBC analytical benchmarks.

**Results (April 2026)**:
- **MSM calibration**: J = 21.79 vs J > 800 uncalibrated (30–40× improvement).
  Calibrated θ\* independently rediscovers literature estimates for US
  matching efficiency (A ≈ 0.98) and Taylor rule sensitivity (φ_π ≈ 1.34).
  7/9 FRED moments have correct sign.
- **Hero IRF**: 10/12 DSGE-correct signs across demand + productivity shocks;
  4/12 statistically significant. Queue goods-clearing mode correctly
  reproduces inflationary sign of demand shock.
- **Total API cost** for all experiments: ≈ $60 on DeepSeek-V3
  ($0.50/simulation × 41 MSM + hero IRF runs + exploratory).

## Setup

Python 3.11+ and a [DeepSeek API key](https://platform.deepseek.com).

```bash
git clone https://github.com/lomalovo/llm-economy-agents.git
cd llm-economy-agents
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env`:

```
DEEPSEEK_API_KEY=sk-...   # required: direct DeepSeek API
FRED_API_KEY=...           # optional: live FRED moments (falls back to hardcoded)
```

Backend is selected in config via `llm.backend_type`:
- `deepseek` — DeepSeek API (default)
- `openai` — OpenAI or compatible
- `dummy` — no API calls, random valid decisions (use for logic testing)

## Quick run

```bash
# Smoke test (free, dummy backend)
python main.py

# Reproduce hero IRF results (~$20, uses calibrated θ* from data/msm/state.json)
python scripts/run_hero_irf.py \
    --theta data/msm/state.json \
    --runs 5 --steps 60 --households 30 \
    --goods-mode queue --out data/hero_queue

# Rebuild IRF charts
HERO_DATA_DIR=data/hero_queue python scripts/compare_irf_full.py

# Full MSM recalibration (~$20, several hours)
python run_msm.py --base config/experiments/msm_base.yaml --lhs 10 --steps 60 --households 13
```

## Architecture

```
config YAML                              data/results/*.csv + *_reasoning.jsonl
    │                                                 ▲
    ▼                                                 │
AgentBuilder → households / firms (LLM)  ──► SimulationEngine.step()
                                             ├── _process_events()      (cash inject / TFP shock)
                                             ├── parallel LLM decisions (asyncio.gather, 13–33 agents)
                                             ├── clear_labor_market     (MP matching + separation rate)
                                             ├── _collect_taxes_and_redistribute
                                             ├── clear_goods_market     (average / queue / weighted)
                                             ├── price update           (Calvo partial adjustment)
                                             ├── _update_interest_rate  (Taylor rule)
                                             ├── _apply_interest_on_savings
                                             └── log_step → CSV + JSONL
```

## Directory map

| Path | Purpose |
|------|---------|
| `config/experiments/` | per-scenario YAMLs (baseline, demand\_shock, productivity\_shock, msm\_base) |
| `src/core/`, `src/agents/`, `src/economics/`, `src/llm/`, `src/analysis/` | core code |
| `scripts/` | pipeline + analysis drivers |
| `tests/` | pytest suite (21 tests) |
| `templates/` | Jinja2 agent prompts |
| `data/msm/` | MSM calibration results (41 evaluations, best θ\* in `state.json`) |
| `data/hero_queue/` | hero IRF CSV files (5 baseline + 5 demand + 5 productivity) |
| `charts/v2/` | paper figures |
| `docs/` | diploma write-up (LaTeX + source markdown) |

## Tests

```bash
python -m pytest tests/ -q   # 21/21 green
```

## Citation

Назаров М. А. «Агентное моделирование экономических систем на основе больших языковых моделей». Дипломная работа, СПбГУ, 2026.
