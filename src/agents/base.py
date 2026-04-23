from abc import ABC, abstractmethod
from src.llm.backend.base import BaseLLMBackend

class BaseAgent(ABC):
    def __init__(self, agent_id: str, llm: BaseLLMBackend,
                 history_window: int = 5, reflection_window: int = 3):
        self.id = agent_id
        self.llm = llm
        self.money: float = 0.0
        self.history_window = history_window
        self.reflection_window = reflection_window
        self.history: list = []
        self.reflections: list = []

    @abstractmethod
    async def make_decision(self, market_data: dict):
        pass

    def get_stats(self) -> dict:
        return {f"{self.id}_money": self.money}

    def _push_history(self, record: dict) -> None:
        self.history.append(record)
        self.history = self.history[-self.history_window:]

    def update_reflection(self, text: str, step: int) -> None:
        self.reflections.append({"step": step, "insight": text.strip()})
        self.reflections = self.reflections[-self.reflection_window:]
