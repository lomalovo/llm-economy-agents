from src.agents.base import BaseAgent
from src.schemas.economics import HouseholdDecision, FirmDecision
from src.llm.prompts import SYSTEM_PROMPT_ECON, HOUSEHOLD_PROMPT, FIRM_PROMPT

class HouseholdAgent(BaseAgent):
    def __init__(self, agent_id, llm, initial_money=100.0):
        super().__init__(agent_id, llm)
        self.money = initial_money
        
        # Последнее решение (Pydantic объект)

        self.labor_hired = 0.0 # Для фирм

        self.current_decision: HouseholdDecision = None 
        
        # Реальные результаты хода (заполняет MarketMechanism)
        self.last_worked: float = 0.0   # Сколько часов реально продал
        self.last_spent: float = 0.0    # Сколько денег реально потратил
        self.last_bought: float = 0.0   # Сколько товаров реально купил

    async def make_decision(self, market_data: dict) -> HouseholdDecision:
        user_msg = HOUSEHOLD_PROMPT.format(
            money=self.money,
            wage=market_data.get("wage", 0),
            price=market_data.get("price", 0)
        )
        
        # Если есть история прошлого хода, можно было бы добавить её в промпт сюда
        # Но пока оставляем базовый вариант
        
        decision = await self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=HouseholdDecision
        )
        
        self.current_decision = decision
        return decision
    
    def get_stats(self) -> dict:
        stats = {
            f"{self.id}_money": round(self.money, 2),
            f"{self.id}_worked": round(self.last_worked, 2),
            f"{self.id}_bought": round(self.last_bought, 2),
        }
        if self.current_decision:
             stats[f"{self.id}_target_spend"] = self.current_decision.consumption_budget
        return stats


class FirmAgent(BaseAgent):
    def __init__(self, agent_id, llm, initial_capital=1000.0):
        super().__init__(agent_id, llm)
        self.money = initial_capital
        self.inventory = 0.0
        
        self.current_decision: FirmDecision = None
        
        # Реальные результаты (заполняет MarketMechanism)
        self.labor_hired: float = 0.0  # Сколько людей реально наняли
        self.goods_sold: float = 0.0   # Сколько товаров реально продали

    async def make_decision(self, market_data: dict) -> FirmDecision:
        user_msg = FIRM_PROMPT.format(
            money=self.money,
            inventory=self.inventory,
            wage=market_data.get("wage", 0),
            last_demand=market_data.get("last_demand", "неизвестно")
        )

        decision = await self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=FirmDecision
        )
        
        self.current_decision = decision
        return decision
