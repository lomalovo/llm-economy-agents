from pydantic import BaseModel
from typing import List

class WorldState(BaseModel):
    step: int = 0
    # Текущие рыночные индикаторы
    avg_wage: float = 15.0
    avg_price: float = 5.0
    total_inventory: float = 0.0
    unemployment_rate: float = 0.0
    
    # История для графиков (можно хранить отдельно, но пока тут)
    history: List[dict] = []
