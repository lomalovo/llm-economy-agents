import asyncio
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
        
        run_name = config.get("experiment", {}).get("name", "sim_run")
        self.logger = SimulationLogger(run_name)
        
        self.households = []
        self.firms = []

    def setup(self):
        agents_cfg = self.cfg.get("agents", {})
        
        # 1. Households
        hh_cfg = agents_cfg.get("households", {"count": 2, "params": {}})
        count = hh_cfg.get("count", 2)
        params = hh_cfg.get("params", {"initial_money": 200})
        
        self.households = [
            HouseholdAgent(f"h_{i+1}", self.llm, **params) 
            for i in range(count)
        ]

        firm_cfg = agents_cfg.get("firms", {"count": 1, "params": {}})
        count = firm_cfg.get("count", 1)
        params = firm_cfg.get("params", {"initial_capital": 1000})
        
        self.firms = [
            FirmAgent(f"f_{i+1}", self.llm, **params) 
            for i in range(count)
        ]
        
        print(f"Setup complete: {len(self.households)} HHs, {len(self.firms)} Firms.")

    async def step(self):
        self.state.step += 1
        print(f"\n--- STEP {self.state.step} START ---")
        
        market_info = {
            "wage": self.state.avg_wage,
            "price": self.state.avg_price,
            "last_demand": 100 # TODO: Сделать динамическим
        }
        
        # --- ФАЗА 1: ПАРАЛЛЕЛЬНОЕ МЫШЛЕНИЕ ---
        print("Agents are thinking (parallel)...")
        # Собираем задачи (tasks)
        tasks = []
        all_agents = self.households + self.firms
        
        for agent in all_agents:
            tasks.append(agent.make_decision(market_info))
            
        # Запускаем их все разом и ждем завершения
        # Если агентов 100, это займет столько же времени, сколько для 1 (если API выдержит)
        results = await asyncio.gather(*tasks)
        
        # --- ФАЗА 2: РЫНКИ (Остается синхронной, т.к. считается локально) ---
        L, emp_rate = self.market.clear_labor_market(self.firms, self.households, self.state.avg_wage)
        self.state.unemployment_rate = 1.0 - emp_rate
        
        sold = self.market.clear_goods_market(self.firms, self.households, self.state.avg_price)
        
        # Обновление цен
        if self.firms:
            new_price = sum(f.current_decision.price_setting for f in self.firms) / len(self.firms)
            self.state.avg_price = new_price
        
        print(f"Stats: Unemp={self.state.unemployment_rate:.0%}, Sold={sold:.1f}, NewPrice={self.state.avg_price:.2f}")

        # --- ЛОГГИРОВАНИЕ ---
        self.logger.log_step(self.state.step, self.state, self.firms, self.households)

    async def run(self, steps=3):
        self.setup()
        try:
            for _ in range(steps):
                await self.step()
                # time.sleep здесь не нужен, если мы хотим скорость
                # Но если упираемся в Rate Limit API: await asyncio.sleep(1)
        finally:
            self.logger.save()