import time
from typing import List
from src.core.state import WorldState
from src.agents.impl import HouseholdAgent, FirmAgent
from src.economics.mechanism import MarketMechanism
from src.llm.backend.base import BaseLLMBackend
from src.core.logger import SimulationLogger

class SimulationEngine:
    def __init__(self, llm: BaseLLMBackend, config: dict):
        self.llm = llm
        self.cfg = config
        self.state = WorldState()
        self.market = MarketMechanism()

        run_name = config.get("experiment", {}).get("name", "experiment")
        self.logger = SimulationLogger(run_name)
        
        self.households: List[HouseholdAgent] = []
        self.firms: List[FirmAgent] = []
        
    def setup(self):
        """Создаем агентов из конфига"""
        # Пока хардкод, позже вынесем в конфиг
        self.households = [
            HouseholdAgent(f"h_{i}", self.llm, initial_money=200) 
            for i in range(2) # 2 домохозяйства
        ]
        self.firms = [
            FirmAgent(f"f_{i}", self.llm, initial_capital=1000) 
            for i in range(1) # 1 фирма
        ]
        print(f"Created {len(self.households)} households and {len(self.firms)} firms.")

    def step(self):
        """Один шаг симуляции (Один месяц/квартал)"""
        self.state.step += 1
        print(f"\n--- STEP {self.state.step} START ---")
        
        # Подготовка данных рынка для агентов
        market_info = {
            "wage": self.state.avg_wage,
            "price": self.state.avg_price,
            "last_demand": 100 # Заглушка
        }
        
        # ФАЗА 1: Планирование (LLM думают параллельно)
        print("Agents are thinking...")
        for agent in self.households + self.firms:
            # Сохраняем решение внутрь агента
            agent.current_decision = agent.make_decision(market_info)
            # Для отладки печатаем одну мысль
            # print(f"[{agent.id}] {agent.current_decision.reasoning[:50]}...")

        # ФАЗА 2: Рынок Труда
        L, employment_rate = self.market.clear_labor_market(
            self.firms, self.households, self.state.avg_wage
        )
        self.state.unemployment_rate = 1.0 - employment_rate
        print(f"Labor Market: Hired {L:.1f}, Unemp Rate: {self.state.unemployment_rate:.1%}")

        # ФАЗА 3: Производство и Рынок Товаров (Цены пока фиксированы для упрощения)
        # В будущем фирмы будут обновлять self.state.avg_price на основе своих решений
        sold_qty = self.market.clear_goods_market(
            self.firms, self.households, self.state.avg_price
        )
        print(f"Goods Market: Sold {sold_qty:.1f} units")
        
        # Обновляем глобальные цены (на основе решения фирмы, если она одна)
        # Если фирм много, считаем среднюю
        new_price_sum = sum(f.current_decision.price_setting for f in self.firms)
        self.state.avg_price = new_price_sum / len(self.firms)
        print(f"New Price set for next step: {self.state.avg_price:.2f}")

        self.logger.log_step(self.state.step, self.state, self.firms, self.households)

    def run(self, steps=3):
        self.setup()
        try:
            for _ in range(steps):
                self.step()
                time.sleep(1)
        finally:
            self.logger.save()

