from typing import Type, Optional
from pydantic import BaseModel
from .base import BaseLLMBackend

class DummyBackend(BaseLLMBackend):
    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema: Optional[Type[BaseModel]] = None
    ):
        print(f"\n[DummyLLM Log]")
        print(f"  System: {system_prompt}")
        print(f"  User:   {user_prompt}")
        
        if schema:
            print(f"  Schema requested: {schema.__name__}")
            # Возвращаем жестко прописанные заглушки для разных схем
            # В будущем здесь можно генерировать рандомные данные библиотекой `polyfactory`
            
            # Если просим TestAgentDecision (из main.py)
            if "TestAgentDecision" in schema.__name__:
                return schema(
                    thoughts="Это тестовый прогон, я имитирую мышление.",
                    action="WAIT",
                    amount=100.50
                )
            
            return schema.model_construct()
            
        return "Dummy text response"
