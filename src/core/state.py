from pydantic import BaseModel

class WorldState(BaseModel):
    step: int = 0
    avg_wage: float = 15.0
    avg_price: float = 5.0
    prev_avg_price: float = 5.0   # previous step's price (for inflation calculation)
    total_inventory: float = 0.0
    unemployment_rate: float = 0.0
    vacancy_rate: float = 0.0     # unfilled labor demand / total supply (Beveridge curve)
    interest_rate: float = 0.05   # Taylor rule output
    last_tax_collected: float = 0.0
    last_redistribution: float = 0.0
