from typing import List
from src.agents.impl import HouseholdAgent, FirmAgent
from src.schemas.economics import HouseholdDecision, FirmDecision

class MarketMechanism:
    def clear_labor_market(self, firms: List[FirmAgent], households: List[HouseholdAgent], wage: float):
        """Сведение спроса и предложения на труд"""
        
        # 1. Сколько всего часов хотят работать люди
        total_supply = sum(h.current_decision.labor_supply for h in households)
        
        # 2. Сколько всего часов хотят купить фирмы
        total_demand = sum(f.current_decision.labor_demand for f in firms)
        
        # 3. Реальный найм (ограничен меньшей стороной)
        actual_labor = min(total_supply, total_demand) if total_supply > 0 and total_demand > 0 else 0
        
        # Коэффициент рационирования (если спрос > предложения, фирмы получат меньше рабочих)
        labor_ratio = actual_labor / total_demand if total_demand > 0 else 0
        # Если работы нет, люди сидят без дела
        employment_ratio = actual_labor / total_supply if total_supply > 0 else 0
        
        # 4. Расчеты
        # Фирмы платят зарплату
        for f in firms:
            hired = f.current_decision.labor_demand * labor_ratio
            cost = hired * wage
            if f.money >= cost:
                f.money -= cost
                f.labor_hired = hired # Запоминаем, сколько реально наняли
            else:
                # Банкротство/Нет денег (упрощенно: наняли 0)
                f.labor_hired = 0
        
        # Люди получают зарплату
        for h in households:
            worked = h.current_decision.labor_supply * employment_ratio
            income = worked * wage
            h.money += income

            h.last_worked = worked
            
        return actual_labor, employment_ratio

    def clear_goods_market(self, firms: List[FirmAgent], households: List[HouseholdAgent], avg_price: float):
        """Сведение рынка товаров"""
        
        # 1. Производство (Упрощенно: Производственная функция Y = L * 1)
        total_production = sum(f.labor_hired * 1.0 for f in firms)
        
        # Добавляем к запасам фирм
        # Для простоты: размажем производство поровну (или пропорционально труду)
        # Пока считаем, что у нас одна мега-фирма или рынок идеален
        
        # 2. Спрос (Сколько денег люди принесли в магазин)
        total_money_supply = sum(h.current_decision.consumption_budget for h in households)
        
        # 3. Сколько реально могут купить товара (Ограничено производством)
        # Если товаров 100, а денег 1000, цена вырастет? Или дефицит?
        # В ABM с фиксированной ценой будет дефицит.
        
        total_goods_available = total_production + sum(f.inventory for f in firms)
        
        max_can_buy = total_money_supply / avg_price if avg_price > 0 else 0
        actual_sold_qty = min(total_goods_available, max_can_buy)
        
        # 4. Обмен
        # Фирмы получают деньги
        revenue = actual_sold_qty * avg_price
        # Распределяем доход между фирмами пропорционально их вкладу (упрощение)
        if total_goods_available > 0:
            for f in firms:
                share = (f.inventory + f.labor_hired) / total_goods_available
                f.money += revenue * share
                f.inventory = (f.inventory + f.labor_hired) - (actual_sold_qty * share)
        
        # Люди получают товар и тратят деньги
        if max_can_buy > 0:
            fill_rate = actual_sold_qty / max_can_buy
            for h in households:
                spent = h.current_decision.consumption_budget * fill_rate
                h.money -= spent

                h.last_spent = spent
                h.last_bought = spent / avg_price if avg_price > 0 else 0
                        
        return actual_sold_qty
