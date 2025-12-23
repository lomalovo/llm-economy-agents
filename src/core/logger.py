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
        
        # Агрегированные данные
        avg_price = world_state.avg_price
        wage = world_state.avg_wage
        unemployment = world_state.unemployment_rate
        
        # ВВП (сумма проданных товаров или произведенных)
        total_production = sum(f.current_decision.production_target for f in firms)
        total_sales = sum(f.goods_sold for f in firms)
        
        # Денежная масса (проверка, не исчезают ли деньги)
        total_money = sum(f.money for f in firms) + sum(h.money for h in households)
        
        row = {
            "step": step,
            "avg_price": avg_price,
            "wage": wage,
            "unemployment_rate": unemployment,
            "total_production": total_production,
            "total_sales": total_sales,
            "total_money_supply": total_money,
            # Можно добавить данные по конкретной фирме, если их мало
            "firm_0_price": firms[0].current_decision.price_setting if firms else 0
        }
        self.data.append(row)

    def save(self):
        """Сохраняем в CSV"""
        if not self.data:
            return
            
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)
        print(f"Data saved to {self.filename}")
        return self.filename
