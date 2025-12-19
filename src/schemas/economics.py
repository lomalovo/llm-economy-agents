from pydantic import BaseModel, Field

# --- Домохозяйство ---
class HouseholdDecision(BaseModel):
    reasoning: str = Field(..., description="Твои размышления: почему ты решил столько работать и тратить?")
    labor_supply: float = Field(..., ge=0.0, le=1.0, description="Доля времени, которую ты готов работать (0.0 - 1.0)")
    consumption_budget: float = Field(..., ge=0.0, description="Сколько денег ты хочешь потратить на товары")
    savings_amount: float = Field(..., description="Сколько денег ты сохраняешь на следующий период")

# --- Фирма ---
class FirmDecision(BaseModel):
    reasoning: str = Field(..., description="Анализ издержек и спроса")
    labor_demand: float = Field(..., ge=0.0, description="Сколько человеко-часов ты хочешь купить")
    price_setting: float = Field(..., ge=0.01, description="Какую цену установить на твой товар")
    production_target: float = Field(..., ge=0.0, description="План производства (штук)")
