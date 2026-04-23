import json
import pandas as pd
from datetime import datetime
from pathlib import Path


class SimulationLogger:
    def __init__(self, run_name: str):
        self.log_dir = Path("data/results")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.filename = self.log_dir / f"{run_name}_{timestamp}.csv"
        self.audit_filename = self.log_dir / f"{run_name}_{timestamp}_reasoning.jsonl"

        self.data = []
        self._audit_lines = []

    def log_step(self, step, world_state, firms, households):
        # Derived DSGE variables
        inflation_rate = (
            (world_state.avg_price - world_state.prev_avg_price) / world_state.prev_avg_price
            if world_state.prev_avg_price > 0 else 0.0
        )
        real_wage = (
            world_state.avg_wage / world_state.avg_price
            if world_state.avg_price > 0 else 0.0
        )
        total_consumption = sum(h.last_spent for h in households)

        row = {
            "step":               step,
            "avg_price":          world_state.avg_price,
            "wage":               world_state.avg_wage,
            "unemployment_rate":  world_state.unemployment_rate,
            "vacancy_rate":       world_state.vacancy_rate,
            "inflation_rate":     round(inflation_rate, 6),
            "real_wage":          round(real_wage, 4),
            "total_consumption":  round(total_consumption, 4),
            "total_production":   sum(f.current_decision.production_target for f in firms if f.current_decision),
            "total_sales":        sum(f.goods_sold for f in firms),
            "total_money_supply": sum(f.money for f in firms) + sum(h.money for h in households),
            "tax_collected":         world_state.last_tax_collected,
            "redistribution_per_hh": world_state.last_redistribution,
            "interest_rate":         world_state.interest_rate,
        }

        for agent in households + firms:
            row.update(agent.get_stats())

        self.data.append(row)

        # Behavioral audit: reasoning + full decision + realized outcomes.
        # Narrative audit joins on this to check reasoning-action consistency.
        for agent in households + firms:
            if not agent.current_decision:
                continue
            dec = agent.current_decision
            record = {
                "step":      step,
                "agent_id":  agent.id,
                "agent_type": "household" if agent in households else "firm",
                "reasoning": dec.reasoning,
                "decision":  dec.model_dump(exclude={"reasoning"}),
            }
            # Realized outcomes so classifier can also assess reasoning vs. result
            if agent in households:
                record["outcome"] = {
                    "worked":  round(agent.last_worked, 4),
                    "spent":   round(agent.last_spent, 4),
                    "bought":  round(agent.last_bought, 4),
                    "money_after": round(agent.money, 2),
                }
            else:
                record["outcome"] = {
                    "hired":     round(agent.labor_hired, 4),
                    "sold":      round(agent.goods_sold, 4),
                    "inventory": round(agent.inventory, 2),
                    "money_after": round(agent.money, 2),
                }
            self._audit_lines.append(record)

    def save(self):
        if not self.data:
            return

        df = pd.DataFrame(self.data)
        cols = ["step"] + [c for c in df.columns if c != "step"]
        df[cols].to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")

        if self._audit_lines:
            with open(self.audit_filename, "w") as f:
                for record in self._audit_lines:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"Reasoning audit saved to {self.audit_filename}")

        return self.filename
