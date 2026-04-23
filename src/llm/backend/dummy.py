import random
from typing import Type, Optional
from pydantic import BaseModel
from .base import BaseLLMBackend


class DummyBackend(BaseLLMBackend):
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Type[BaseModel]] = None
    ):
        if schema:
            name = schema.__name__

            if name == "HouseholdDecision":
                labor   = round(random.uniform(0.5, 0.95), 2)
                budget  = round(random.uniform(10.0, 60.0), 2)
                savings = round(random.uniform(5.0, 30.0), 2)
                return schema(
                    reasoning="Dummy: balancing current needs against saving for later.",
                    labor_supply=labor,
                    consumption_budget=budget,
                    savings_amount=savings,
                )

            if name == "FirmDecision":
                return schema(
                    reasoning="Dummy: assessing demand and setting a competitive price.",
                    labor_demand=round(random.uniform(1.0, 5.0), 2),
                    price_setting=round(random.uniform(10.0, 25.0), 2),
                    production_target=round(random.uniform(1.0, 8.0), 2),
                )

            # Fallback for unknown schemas
            try:
                fields = {k: 0.0 for k in schema.model_fields}
                return schema(**fields)
            except Exception:
                return schema.model_construct()

        # No schema → free text (reflection)
        return "Dummy reflection: conditions are stable, maintaining current strategy."
