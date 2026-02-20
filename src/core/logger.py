import csv
import pandas as pd
from datetime import datetime
from pathlib import Path

class SimulationLogger:
    def __init__(self, run_name: str):
        # Создаем папку для логов, если нет
        self.log_dir = Path("data/results")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.log_dir / f"{run_name}_{timestamp}.csv"
        
        # Буфер для данных
        self.data = []
        
    def log_step(self, step, world_state, firms, households):
        """Собираем метрики со всего мира в одну строку"""
        
        # 1. Агрегированные данные (Макро)
        row = {
            "step": step,
            "avg_price": world_state.avg_price,
            "wage": world_state.avg_wage,
            "unemployment_rate": world_state.unemployment_rate,
            "total_production": sum(f.current_decision.production_target for f in firms if f.current_decision),
            "total_sales": sum(f.goods_sold for f in firms),
            "total_money_supply": sum(f.money for f in firms) + sum(h.money for h in households),
        }

        # 2. Данные по КАЖДОМУ агенту (Микро)
        # Мы проходимся по списку агентов и сливаем их stats в общую строку
        for agent in households + firms:
            # agent.get_stats() возвращает словарь типа {"savers_1_money": 200, ...}
            stats = agent.get_stats()
            row.update(stats)

        self.data.append(row)

    def save(self):
        """Сохраняем в CSV"""
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        cols = ["step"] + [c for c in df.columns if c != "step"]
        df = df[cols]
        
        df.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")
        return self.filename
