from pydantic import BaseModel

class TestAgentDecision(BaseModel):
    thoughts: str
    action: str
    amount: float
