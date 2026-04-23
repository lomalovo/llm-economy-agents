from typing import Optional, cast
from src.agents.base import BaseAgent
from src.schemas.economics import HouseholdDecision, FirmDecision
from src.llm.prompt_manager import SYSTEM_PROMPT_ECON, get_prompt_manager

class HouseholdAgent(BaseAgent):
    def __init__(self, agent_id, llm, initial_money=100.0, template="household.j2",
                 history_window=5, reflection_window=3, **custom_params):
        super().__init__(agent_id, llm, history_window=history_window,
                         reflection_window=reflection_window)
        self.money = initial_money
        self.template_name = template
        self.attributes = custom_params

        self.current_decision: Optional[HouseholdDecision] = None
        self.pending_event_notifications: list[str] = []

        self.last_worked: float = 0.0
        self.last_spent: float = 0.0
        self.last_bought: float = 0.0
        self.last_tax_paid: float = 0.0
        self.last_redistribution: float = 0.0
        self.last_savings: float = 0.0
        self.last_interest_earned: float = 0.0

    def update_history(self, step: int, wage: float):
        self._push_history({
            "step":           step,
            "worked":         round(self.last_worked, 2),
            "earned":         round(self.last_worked * wage, 2),
            "tax_paid":       round(self.last_tax_paid, 2),
            "redistribution": round(self.last_redistribution, 2),
            "spent":          round(self.last_spent, 2),
            "bought":         round(self.last_bought, 2),
            "money":          round(self.money, 2),
        })

    async def make_decision(self, market_data: dict) -> HouseholdDecision:
        notifications = self.pending_event_notifications[:]
        self.pending_event_notifications = []

        context = {
            "agent_id":             self.id,
            "money":                self.money,
            "wage":                 market_data.get("wage", 0),
            "price":                market_data.get("price", 0),
            "history":              self.history,
            "reflections":          self.reflections,
            "event_notifications":  notifications,
            **market_data,
            **self.attributes
        }

        prompter = get_prompt_manager()
        user_msg = prompter.render(self.template_name, **context)

        decision = cast(HouseholdDecision, await self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=HouseholdDecision
        ))

        self.current_decision = decision
        return decision

    def get_stats(self) -> dict:
        stats = {
            f"{self.id}_money":        round(self.money, 2),
            f"{self.id}_worked":       round(self.last_worked, 2),
            f"{self.id}_bought":       round(self.last_bought, 2),
        }
        if self.current_decision:
            stats[f"{self.id}_target_spend"] = self.current_decision.consumption_budget
        return stats


class FirmAgent(BaseAgent):
    def __init__(self, agent_id, llm, initial_capital=1000.0, template="firm.j2",
                 history_window=5, reflection_window=3, productivity=1.0, **custom_params):
        super().__init__(agent_id, llm, history_window=history_window,
                         reflection_window=reflection_window)
        self.money = initial_capital
        self.inventory = 0.0
        self.productivity = productivity
        self.template_name = template
        self.attributes = custom_params

        self.current_decision: Optional[FirmDecision] = None
        self.pending_event_notifications: list[str] = []

        self.labor_hired: float = 0.0
        self.goods_sold: float = 0.0

    def update_history(self, step: int):
        self._push_history({
            "step":      step,
            "hired":     round(self.labor_hired, 2),
            "sold":      round(self.goods_sold, 2),
            "inventory": round(self.inventory, 2),
            "money":     round(self.money, 2),
            "price_set": round(self.current_decision.price_setting, 2) if self.current_decision else 0.0,
        })

    def get_stats(self) -> dict:
        stats = {
            f"{self.id}_money":     round(self.money, 2),
            f"{self.id}_inventory": round(self.inventory, 2),
            f"{self.id}_sold":      round(self.goods_sold, 2),
        }
        if self.current_decision:
            stats[f"{self.id}_price"] = self.current_decision.price_setting
        return stats

    async def make_decision(self, market_data: dict) -> FirmDecision:
        notifications = self.pending_event_notifications[:]
        self.pending_event_notifications = []

        context = {
            "agent_id":             self.id,
            "step":                 market_data.get("step", 0),
            "money":                self.money,
            "inventory":            self.inventory,
            "productivity":         self.productivity,
            "wage":                 market_data.get("wage", 0),
            "last_sales":           self.goods_sold,
            "last_demand":          market_data.get("last_demand", "unknown"),
            "total_consumer_demand": market_data.get("total_consumer_demand", "unknown"),
            "sell_ratio":           market_data.get("sell_ratio", "unknown"),
            "unemployment_rate":    market_data.get("unemployment_rate", 0),
            "vacancy_rate":         market_data.get("vacancy_rate", 0),
            "inflation_rate":       market_data.get("inflation_rate", 0),
            "interest_rate":        market_data.get("interest_rate", 0),
            "history":              self.history,
            "reflections":          self.reflections,
            "event_notifications":  notifications,
            **self.attributes
        }

        prompter = get_prompt_manager()
        user_msg = prompter.render(self.template_name, **context)

        decision = cast(FirmDecision, await self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=FirmDecision
        ))

        self.current_decision = decision
        return decision
