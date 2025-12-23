from src.llm.backend.base import BaseLLMBackend
from src.llm.backend.dummy import DummyBackend
from src.llm.backend.openai_api import OpenAICompatibleBackend

def get_llm_backend(config: dict) -> BaseLLMBackend:
    llm_cfg = config.get("llm", {}) # Берем секцию целиком
    
    backend_type = llm_cfg.get("backend_type", "dummy").lower()
    
    max_concurrency = llm_cfg.get("max_concurrency", 5)
    max_retries = llm_cfg.get("max_retries", 3)
    
    if backend_type == "dummy":
        return DummyBackend()
    
    elif backend_type == "deepseek":
        return OpenAICompatibleBackend(
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            model_name=llm_cfg.get("model_name", "deepseek-chat"),
            # Прокидываем лимиты
            max_concurrency=max_concurrency,
            max_retries=max_retries
        )
        
    elif backend_type == "openai":
         return OpenAICompatibleBackend(
            api_key_env_var="OPENAI_API_KEY",
            base_url=None,
            model_name=llm_cfg.get("model_name", "gpt-3.5-turbo"),
            max_concurrency=max_concurrency,
            max_retries=max_retries
        )

    # ... место для HuggingFace ...
        
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
