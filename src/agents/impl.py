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

    def make_decision(self, market_data: dict) -> HouseholdDecision:
        user_msg = HOUSEHOLD_PROMPT.format(
            money=self.money,
            wage=market_data.get("wage", 0),
            price=market_data.get("price", 0)
        )
        
        # Если есть история прошлого хода, можно было бы добавить её в промпт сюда
        # Но пока оставляем базовый вариант
        
        decision = self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=HouseholdDecision
        )
        
        # Важно: Сохраняем решение в self
        self.current_decision = decision
        return decision


class FirmAgent(BaseAgent):
    def __init__(self, agent_id, llm, initial_capital=1000.0):
        super().__init__(agent_id, llm)
        self.money = initial_capital
        self.inventory = 0.0
        
        self.current_decision: FirmDecision = None
        
        # Реальные результаты (заполняет MarketMechanism)
        self.labor_hired: float = 0.0  # Сколько людей реально наняли
        self.goods_sold: float = 0.0   # Сколько товаров реально продали

    def make_decision(self, market_data: dict) -> FirmDecision:
        user_msg = FIRM_PROMPT.format(
            money=self.money,
            inventory=self.inventory,
            wage=market_data.get("wage", 0),
            last_demand=market_data.get("last_demand", "неизвестно")
        )

        decision = self.llm.generate(
            system_prompt=SYSTEM_PROMPT_ECON,
            user_prompt=user_msg,
            schema=FirmDecision
        )
        
        self.current_decision = decision
        return decision
