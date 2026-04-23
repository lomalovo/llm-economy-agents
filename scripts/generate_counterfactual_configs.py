"""Generate 4 counterfactual bio configs from the shared bios file.

Conditions:
    all_htm     — 10 HH, cycling 3 hand-to-mouth bios
    all_saver   — 10 HH, cycling 3 saver bios
    mixed       — 10 HH, unique heterogeneous bios (full diversity)
    fifty_fifty — 10 HH, explicit 5 HtM + 5 saver
"""
from __future__ import annotations
import yaml
from pathlib import Path
from copy import deepcopy


BIOS_FILE = Path(__file__).resolve().parents[1] / "data" / "bios_library.yaml"

# Unique 10-bio mix for the "mixed" condition
MIXED_INDICES = list(range(10))

FIRM_BIOS = [
    (
        "Your company is a mid-size components manufacturer that has competed on "
        "price since its founding. The original owner built the business by being "
        "cheaper than anyone else and that logic has been inherited by everyone "
        "who followed. The operation is lean by design: low overhead, high volume, "
        "a workforce that understands it will be adjusted as demand dictates. "
        "Margins are thin and intentionally so — the model only works at scale. "
        "There is no premium line, no aspirations to develop one."
    ),
    (
        "Your company has been producing consumer goods for over twenty years and "
        "has built a reputation among its regular buyers for consistency and "
        "reliability. It has never been the cheapest option in its segment and has "
        "never tried to be. Its customers accept a modest price premium in exchange "
        "for dependable quality and predictable availability. The firm employs "
        "skilled workers and treats retention as a strategic priority."
    ),
    (
        "Your company is three years old and still in the market-building phase. "
        "It was founded by two partners who believed the market was underdeveloped "
        "and that the window to gain position would not stay open for long. The "
        "strategy since day one has been aggressive: accept lower margins than "
        "competitors, move volume, establish presence. The workforce is young and "
        "relatively inexpensive."
    ),
]


# Full heterogeneous 10-bio mix for "mixed" — 3 HtM + 4 middle + 3 saver
MIDDLE_BIOS = [
    "You are Sofia, 26, the first in your family to graduate from university. Your parents were not poor but money was always 'careful' at home. You are eighteen months into your first analyst job and genuinely enjoying having a salary for the first time. You set up an automatic bank transfer each month mostly to feel like a responsible adult — not because you have a concrete savings goal.",
    "You are Oleg, 44, a logistics manager who has worked in the same industry for twenty years. Your salary is stable, your car is paid off, your two children's school fees are manageable. You think of yourself as someone who is reasonably good with money, though you would struggle to explain exactly why — things have generally worked out and you have not been pushed to examine the reasons closely.",
    "You are Lena, 32, a freelance graphic designer self-employed for six years. Your earnings arrive in violent lurches: a large invoice clears and you feel briefly wealthy; then nothing comes in for weeks and you feel the familiar tightening. You have invented a financial system for yourself over the years. You have never fully managed to stick to it.",
    "You are Pavel, 51, a secondary-school physics teacher at the same school for eighteen years. Your salary is modest and predictable. You are not interested in luxuries. Your pleasures are cheap ones: books, long walks, cooking at home. You worry about retirement in a vague way but looking at the actual numbers makes the worry more concrete and harder to live with, so you avoid doing it.",
]


def build_base_cfg(experiment_name: str, hh_bios: list[str], firm_bios: list[str]) -> dict:
    return {
        "experiment": {
            "name": experiment_name,
            "description": f"Counterfactual bio composition: {experiment_name}",
        },
        "initial_state": {
            "avg_price": 15.0,
            "prev_avg_price": 15.0,
            "avg_wage": 15.0,
        },
        "market": {
            "goods_clearing_mode": "average",
            "wage_adjustment_speed": 0.15,
            "wage_max_increase": 0.10,
            "wage_max_decrease": 0.05,
            "matching_efficiency": 0.6,
            "matching_elasticity": 0.5,
            "separation_rate": 0.05,
            "price_adjustment_speed": 0.4,
        },
        "simulation": {
            "history_window": 5,
            "reflection_every": 1000,   # disable reflections to save tokens
            "reflection_window": 3,
        },
        "central_bank": {
            "enabled": True,
            "neutral_rate": 0.05,
            "target_inflation": 0.02,
            "target_unemployment": 0.05,
            "inflation_sensitivity": 1.5,
            "unemployment_sensitivity": 0.5,
            "min_rate": 0.0,
            "max_rate": 0.25,
        },
        "government": {
            "enabled": True,
            "tax_brackets": [
                {"threshold": 0, "rate": 0.10},
                {"threshold": 20, "rate": 0.20},
                {"threshold": 50, "rate": 0.35},
            ],
        },
        "llm": {
            "backend_type": "eliza",
            "model_name": "deepseek/deepseek-chat-v3-0324",
            "temperature": 0.7,
            "max_concurrency": 30,
            "max_retries": 3,
            "timeout": 60,
        },
        "events": [
            {
                "step": 10,
                "type": "cash_injection",
                "target": "household",
                "amount": 100.0,
                "description": "Helicopter money — demand shock",
            }
        ],
        "agents": {
            "households": {
                "type": "household",
                "count": len(hh_bios),
                "params": {
                    "initial_money": 200.0,
                    "template": "household.j2",
                    "bios": hh_bios,
                },
            },
            "firms": {
                "type": "firm",
                "count": len(firm_bios),
                "params": {
                    "initial_capital": 1000.0,
                    "template": "firm.j2",
                    "bios": firm_bios,
                },
            },
        },
    }


def main():
    with open(BIOS_FILE) as f:
        lib = yaml.safe_load(f)

    htm = [b.strip() for b in lib["htm"]]
    saver = [b.strip() for b in lib["saver"]]

    # Need exactly 10 entries for each condition
    def take(bios, n):
        result = []
        for i in range(n):
            result.append(bios[i % len(bios)])
        return result

    conditions = {
        "cf_all_htm":      take(htm, 10),
        "cf_all_saver":    take(saver, 10),
        "cf_mixed":        htm + MIDDLE_BIOS + saver,   # 3 + 4 + 3 = 10
        "cf_fifty_fifty":  htm + htm[:2] + saver + saver[:2],  # 3+2+3+2 = 10
    }

    out_dir = Path("config/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, hh_bios in conditions.items():
        assert len(hh_bios) == 10, f"{name} has {len(hh_bios)} bios, expected 10"
        cfg = build_base_cfg(name, hh_bios, FIRM_BIOS)
        out_path = out_dir / f"{name}.yaml"
        with open(out_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, width=100)
        print(f"Wrote {out_path} (10 households: {' | '.join(hh_bios[0].split()[2:4] + hh_bios[-1].split()[2:4])})")


if __name__ == "__main__":
    main()
