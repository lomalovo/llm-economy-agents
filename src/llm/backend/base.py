from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

class BaseLLMBackend(ABC):
    @abstractmethod
    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema: Optional[Type[T]] = None
    ) -> T | str:
        """
        - system_prompt: Роль агента (Ты эконом. агент...)
        - user_prompt: Входящие данные (Цены такие-то, что делаешь?)
        - schema: Класс Pydantic, в который надо распарсить ответ.
        """
        pass
