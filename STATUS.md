# Session checkpoint — 2026-05-15 10:30 MSK

## TL;DR где мы

- **Architecture**: новая (`history_window=10`, `goods_clearing_mode=weighted` для hero, `average` для MSM osnovy)
- **Best θ\* candidates** определены через multi-criteria scoring (J + IRF quality)
- **Winner 1 (bo_000)** — hero IRF полностью готов (15/15 runs)
- **Winner 2 (lhs2_011)** — остановлен на baseline 5/5 (demand/productivity не запускались по решению пользователя)
- **Финальные графики w1 не построены** — pipeline остановлен до этого шага

---

## Что сделано (полностью)

### MSM-калибровка (под live FRED, 1990-2024)
- 41 evaluation total: 10 LHS seed=42 + 16 LHS seed=43 + 5 BO + 10 perturbations
- Top-5 по combined J + IRF quality:
  1. **bo_000** combined=0.998 — A=0.998, price_adj=0.994, phi_pi=2.08, J=26.48, IRFq=5.56
  2. **lhs2_011** combined=0.970 — A=0.899, price_adj=0.988, phi_pi=1.91, J=44.23, IRFq=5.44
  3. **pert_sticky_02** combined=0.917 — A=0.913, price_adj=0.162, phi_pi=1.80, J=58.63, IRFq=5.17
  4. **lhs_002** combined=0.915 — A=0.916, price_adj=0.971, phi_pi=1.45, J=24.77, IRFq=5.07
  5. **pert_sticky_00** combined=0.906 — A=0.968, price_adj=0.196, phi_pi=2.01, J=42.57, IRFq=5.06

### Refinement (5 runs at lhs2_004 = old best, average mode, history=5)
- refine_00..04: J range 21–70, median 23
- Best refine: 20.97 (refine_01)

### Hero IRF: Winner 1 (bo_000)
- `data/hero_weighted_w1/`: 5 baseline + 5 demand_shock + 5 productivity_shock CSVs (всё ✓)
- Параметры runs: N=30, 60 steps, goods_mode=weighted, history_window=10
- **`data/msm/state_winner1.json`** — θ saved

### Counterfactual + narrative audit + scale ablation
- Не пересмотрены под NEW arch — остались данные от предыдущих фаз
- Counterfactual: `data/counterfactual/` (под `average` + history=5)
- Audit: `data/audit/judgments.jsonl` + summary.json (2598 records, 50.3% consistency)
- Scale: `data/scale_ablation/scale_summary.csv` (на dummy)

---

## A=0.998 vs theory ≈0.9 (открытое решение)

Концептуальный gap: MSM выкручивает A на upper bound (0.998), теория (Shimer 2005) ожидает 0.9. **lhs2_011 (winner 2)** имеет A=0.899 — theory-consistent, но combined score чуть хуже.

Не решено для paper: positioning bo_000 (best-fit) vs lhs2_011 (theory-aligned). Винeр 2 hero не успел запуститься (остановлен в demand_shock). Если решим показать оба — нужно довести winner 2.

---

## Что НЕ сделано / pending

### 1. Compare_irf_dsge для Winner 1
Команда:
```bash
HERO_DATA_DIR=data/hero_weighted_w1 python scripts/compare_irf_dsge.py
mv -f charts/v2/fig_irf_vs_dsge_demand.png charts/v2/fig_irf_vs_dsge_demand_w1.png
mv -f charts/v2/fig_irf_vs_rbc_productivity.png charts/v2/fig_irf_vs_rbc_productivity_w1.png
```

### 2. Final build (results table + charts)
```bash
python scripts/msm_full_analysis.py
python scripts/build_results_table.py
python scripts/make_paper_charts.py
```

### 3. Winner 2 hero IRF (опционально — если хотим оба θ в paper)
Текущее состояние:
- `data/hero_weighted_w2/baseline_run_0..4.csv` (5/5 done)
- `data/hero_weighted_w2/progress.jsonl`
- demand_shock + productivity_shock missing

Resume команда:
```bash
python scripts/run_hero_irf.py \
  --theta data/msm/state_winner2.json \
  --runs 5 --steps 60 --households 30 \
  --goods-mode weighted \
  --out data/hero_weighted_w2
```
Resume-aware: пропустит готовые 5 baseline, запустит demand+productivity (~1.5h, ~$15).

### 4. PAPER.md update (Section 5.3, 5.4)
- Подменить старые demand/productivity IRF на новые (под winner 1 + weighted)
- Добавить discussion про A=0.998 (upper bound vs theory)
- Если winner 2 готов — добавить как robust alternative

---

## Файлы на диске (важные)

```
data/msm/
├── lhs_results.jsonl          # 41 evaluations
├── refine_results.jsonl       # 5 refinement runs
├── queue_validation.jsonl     # 3 θ × queue mode test
├── failures.jsonl             # API outage failures (cleanup OK)
├── state.json                 # best θ overall (lhs2_004, old paradigm)
├── state_winner1.json         # bo_000 (NEW best by combined score)
├── state_winner2.json         # lhs2_011 (theory-A alternative)
├── theta_ranking.json         # top-15 by combined score
└── lhs_summary.csv

data/hero_weighted_w1/          # bo_000, complete
├── baseline_run_0..4.csv
├── demand_shock_run_0..4.csv
├── productivity_shock_run_0..4.csv
└── progress.jsonl

data/hero_weighted_w2/          # lhs2_011, PARTIAL
├── baseline_run_0..4.csv
└── progress.jsonl
# missing: demand_shock_run_*, productivity_shock_run_*

charts/v2/
├── fig_msm_J.png + fig_msm_A_pattern.png + fig_moment_fit.png
├── fig_irf_vs_dsge_demand.png + fig_irf_vs_rbc_productivity.png (OLD — from before perturbation)
├── fig_scale.png, fig_counterfactual.png, fig_narrative.png
# missing: fig_irf_vs_dsge_demand_w1.png, fig_irf_vs_rbc_productivity_w1.png (need re-run)

data/final_results.md + data/final_results.json   # stale — нужно build_results_table
```

---

## Resume — точные команды

После рестарта терминала с новым токеном, чтобы доделать минимум:

```bash
cd /Users/lomalovo/Documents/llm-economy-agents
source venv/bin/activate

# 1. Compare IRF: winner 1 vs DSGE/RBC
HERO_DATA_DIR=data/hero_weighted_w1 python scripts/compare_irf_dsge.py
mv -f charts/v2/fig_irf_vs_dsge_demand.png charts/v2/fig_irf_vs_dsge_demand_w1.png
mv -f charts/v2/fig_irf_vs_rbc_productivity.png charts/v2/fig_irf_vs_rbc_productivity_w1.png

# 2. Final build
python scripts/msm_full_analysis.py
python scripts/build_results_table.py
python scripts/make_paper_charts.py

# Result:
# - charts/v2/fig_irf_vs_dsge_demand_w1.png  (main IRF for paper)
# - data/final_results.md (updated)
# - data/msm/state.json (re-pointed to bo_000 if msm_full_analysis picks by J alone)
```

Optional (если хочешь winner 2 как alternative):
```bash
# Resume winner 2 hero (~1.5h, ~$15)
python scripts/run_hero_irf.py --theta data/msm/state_winner2.json \
    --runs 5 --steps 60 --households 30 \
    --goods-mode weighted --out data/hero_weighted_w2

# Compare for w2
HERO_DATA_DIR=data/hero_weighted_w2 python scripts/compare_irf_dsge.py
mv -f charts/v2/fig_irf_vs_dsge_demand.png charts/v2/fig_irf_vs_dsge_demand_w2.png
mv -f charts/v2/fig_irf_vs_rbc_productivity.png charts/v2/fig_irf_vs_rbc_productivity_w2.png
```

---

## Бюджет

Потрачено за всю сессию: **~$60-70** (точно не знаю — много API outage failures).

---

## Открытые архитектурные вопросы для обсуждения в paper

1. **A=0.998 в bo_000 vs theory A=0.9** — какой θ ставим primary?
   - bo_000 (combined 0.998) — best-fit
   - lhs2_011 (combined 0.970) — theory-aligned
2. **`history_window=10` vs old 5** — улучшает ли IRF shapes?
3. **`weighted` vs `queue` vs `average`** для goods market — какой даёт лучший Phillips/inflation IRF?
4. **Inflation IRF всё ещё плоская под demand shock** — `MARKET SIGNAL` notification в firm prompt не достаточен. Нужен ли prompt-edit (но user сказал "никаких инструкций в промпте")?

---

## Известные баги

- `compare_irf_dsge.py`: при отсутствии входных CSV выпадает с ValueError. Использовать только когда `data/hero_weighted_w{1,2}/` complete.
- `bash scripts/orchestrate*.sh` загружает в память один раз — изменения в файле во время выполнения НЕ применяются. Если меняешь скрипт — нужен kill + restart.
- `zsh glob` `data/foo/*` падает если ничего не матчится. Использовать exact filenames для `rm`.

---

## Quick health check после рестарта

```bash
# 1. Tests
python -m pytest tests/ -q  # должно быть 21/21 green

# 2. API alive?
python -c "
import asyncio
from src.llm.factory import get_llm_backend
from src.schemas.economics import HouseholdDecision
async def t():
    llm = get_llm_backend({'llm': {'backend_type': 'eliza', 'model_name': 'deepseek/deepseek-chat-v3-0324', 'max_concurrency': 3, 'max_retries': 2, 'timeout': 20}})
    try:
        await llm.generate('test','set labor_supply=0.5,consumption_budget=30,savings_amount=20', schema=HouseholdDecision)
        print('API OK')
    except Exception as e:
        print('API DOWN:', e)
asyncio.run(t())
"

# 3. Что готово на диске?
ls data/hero_weighted_w1/  # 16 файлов (15 CSV + 1 progress)
ls data/msm/state_winner*.json  # 2 файла
cat data/msm/theta_ranking.json | python3 -c "import json,sys; r=json.load(sys.stdin); print(f'Top 3: {[x[chr(34)+chr(108)+chr(97)+chr(98)+chr(101)+chr(108)+chr(34)] for x in r[:3]]}')"
```
