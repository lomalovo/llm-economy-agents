from src.llm.backend.base import BaseLLMBackend
from src.llm.backend.dummy import DummyBackend

# from src.llm.backend.openai_api import OpenAIBackend
# from src.llm.backend.huggingface import HuggingFaceBackend

def get_llm_backend(config: dict) -> BaseLLMBackend:
    """
    Создает экземпляр LLM бэкенда на основе конфига.
    """
    backend_type = config["llm"]["backend_type"].lower()
    
    if backend_type == "dummy":
        return DummyBackend()
    
    elif backend_type == "openai":
        # return OpenAIBackend(api_key=..., model=config["llm"]["model_name"])
        raise NotImplementedError("OpenAI backend not implemented yet")
        
    elif backend_type == "huggingface":
        # return HuggingFaceBackend(model_name=config["llm"]["model_name"])
        raise NotImplementedError("HuggingFace backend not implemented yet")
        
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
