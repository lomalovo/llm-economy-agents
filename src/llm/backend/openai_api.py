import os
import json
import asyncio
import random
from typing import Type, Optional
from pydantic import BaseModel
from openai import AsyncOpenAI, APITimeoutError, APIStatusError
from dotenv import load_dotenv

from .base import BaseLLMBackend
from src.llm.parsing import extract_json_from_text

load_dotenv()

class OpenAICompatibleBackend(BaseLLMBackend):
    def __init__(
        self, 
        api_key_env_var: str, 
        base_url: str = None, 
        model_name: str = "gpt-3.5-turbo",
        max_concurrency: int = 5,
        max_retries: int = 3,
        timeout: int = 60
    ):
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise ValueError(f"API Key env var '{api_key_env_var}' is missing/empty!")

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        self.model_name = model_name
        self.max_retries = max_retries
        
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        schema: Optional[Type[BaseModel]] = None
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if schema:
            json_structure = json.dumps(schema.model_json_schema(), indent=2)
            instr = (
                f"\n\nOUTPUT FORMAT INSTRUCTION:\n"
                f"1. You MUST respond with a valid JSON object matching this schema:\n{json_structure}\n"
                f"2. Do NOT wrap in markdown blocks.\n"
                f"3. IMPORTANT: No math expressions (e.g., '50*10'). Calculate values yourself and output ONLY numbers."
            )
            messages[0]["content"] += instr

        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.7
                    )
                
                content = response.choices[0].message.content
                
                if schema:
                    clean_json = extract_json_from_text(content)
                    return schema.model_validate_json(clean_json)
                
                return content

            except (APITimeoutError, APIStatusError) as e:
                last_exception = e
                if attempt == self.max_retries:
                    print(f"[LLM Error] Final attempt failed: {e}")
                    raise e
                
                # Exponential Backoff + Jitter
                # Ждем: 1с, 2с, 4с... + случайная добавка (чтобы все агенты не ломанулись разом снова)
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[LLM Warning] Request failed ({e}). Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                print(f"[Code Error] Unexpected error: {e}")
                raise e
