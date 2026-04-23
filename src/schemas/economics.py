import re
from pydantic import BaseModel, Field, field_validator


def _to_float(v, default: float = 0.0) -> float:
    """Robustly convert LLM output to float. Handles dicts, lists, roman numerals, expressions."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        # Take the first numeric element
        for item in v:
            if isinstance(item, (int, float)):
                return float(item)
            if isinstance(item, str):
                try:
                    return _to_float(item, default)
                except Exception:
                    pass
        return default
    if isinstance(v, dict):
        # e.g. {"value": 10.5} or {"price": 10.5}
        for key in ("value", "price", "amount", "result"):
            if key in v:
                return _to_float(v[key], default)
        # Take the first numeric value found
        for val in v.values():
            if isinstance(val, (int, float)):
                return float(val)
        return default
    if isinstance(v, str):
        v = v.strip()
        # Extract first number (including decimals)
        match = re.search(r"-?\d+(?:\.\d+)?", v)
        if match:
            return float(match.group())
        return default
    return default


class HouseholdDecision(BaseModel):
    reasoning: str = Field(..., description="Your reasoning: why did you decide to work and spend this much?")
    labor_supply: float = Field(..., ge=0.0, le=1.0, description="Fraction of time you are willing to work (0.0–1.0)")
    consumption_budget: float = Field(..., ge=0.0, description="How much money you want to spend on goods")
    savings_amount: float = Field(..., description="How much money you save for the next period")

    @field_validator("labor_supply", mode="before")
    @classmethod
    def clamp_labor_supply(cls, v):
        return max(0.0, min(1.0, _to_float(v, 0.5)))

    @field_validator("consumption_budget", "savings_amount", mode="before")
    @classmethod
    def clamp_non_negative(cls, v):
        return max(0.0, _to_float(v, 0.0))


class FirmDecision(BaseModel):
    reasoning: str = Field(..., description="Your cost and demand analysis")
    labor_demand: float = Field(..., ge=0.0, description="How many labor-hours you want to hire")
    price_setting: float = Field(..., ge=0.01, description="What price to set for your product")
    production_target: float = Field(..., ge=0.0, description="Production plan (units)")

    @field_validator("labor_demand", "production_target", mode="before")
    @classmethod
    def clamp_non_negative_firm(cls, v):
        return max(0.0, _to_float(v, 0.0))

    @field_validator("price_setting", mode="before")
    @classmethod
    def clamp_price(cls, v):
        return max(0.01, _to_float(v, 1.0))
