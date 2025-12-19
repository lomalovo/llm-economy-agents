from abc import ABC, abstractmethod
from src.llm.backend.base import BaseLLMBackend

class BaseAgent(ABC):
    def __init__(self, agent_id: str, llm: BaseLLMBackend):
        self.id = agent_id
        self.llm = llm
        self.money: float = 0.0

    @abstractmethod
    def make_decision(self, market_data: dict):
        pass
