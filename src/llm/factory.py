from src.llm.backend.base import BaseLLMBackend
from src.llm.backend.dummy import DummyBackend
from src.llm.backend.openai_api import OpenAICompatibleBackend

def get_llm_backend(config: dict) -> BaseLLMBackend:
    backend_type = config["llm"]["backend_type"].lower()
    
    if backend_type == "dummy":
        return DummyBackend()
    
    elif backend_type == "deepseek":
        return OpenAICompatibleBackend(
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            model_name=config["llm"]["model_name"]
        )
        
    elif backend_type == "openai":
        return OpenAICompatibleBackend(
            api_key_env_var="OPENAI_API_KEY",
            base_url=None,
            model_name=config["llm"]["model_name"]
        )

    # ... место для HuggingFace ...
        
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
