import os
import json
from typing import Type, Optional, cast
from pydantic import BaseModel
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv

from .base import BaseLLMBackend
from src.llm.parsing import extract_json_from_text

# Загружаем переменные из .env
load_dotenv()

class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(self, api_key_env_var: str, base_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        :param api_key_env_var: Имя переменной окружения
        :param base_url: Ссылка на API
        """
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            print(f"Warning: API Key {api_key_env_var} not found in .env")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name

    def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema: Optional[Type[BaseModel]] = None
    ):
        system_content = system_prompt
        if schema:
            json_structure = json.dumps(schema.model_json_schema(), indent=2)
            instr = (
                f"\n\nВАЖНО: Ответ должен быть строго в формате JSON, соответствующем схеме:\n"
                f"{json_structure}\n"
                f"Не пиши никаких пояснений, только JSON."
            )
            system_content += instr
        
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            if content is None:
                raise ValueError("API returned empty response")
            
            if schema:
                clean_json = extract_json_from_text(content)
                return schema.model_validate_json(clean_json)
            
            return content

        except Exception as e:
            # Тут в будущем добавим повторные попытки (retries)
            print(f"API Error: {e}")
            print(f"Raw content was: {locals().get('content', 'No content')}")
            raise e
