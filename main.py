import sys
from src.utils.config import load_config
from src.llm.factory import get_llm_backend
from src.schemas.test_schema import TestAgentDecision

def main():
    try:
        cfg = load_config()
        print(f"Конфигурация загружена: {cfg['experiment']['name']}")
    except Exception as e:
        print(f"Ошибка загрузки конфига: {e}")
        sys.exit(1)

    try:
        llm = get_llm_backend(cfg)
        print(f"LLM Backend инициализирован: {type(llm).__name__}")
    except Exception as e:
        print(f"Ошибка инициализации LLM: {e}")
        sys.exit(1)

    print("\n--- ЗАПУСК ТЕСТОВОГО ЗАПРОСА ---")
    
    try:
        decision = llm.generate(
            system_prompt="Ты экономический агент.",
            user_prompt="Цена на хлеб выросла. Что будешь делать?",
            schema=TestAgentDecision 
        )
        
        print("\n--- РЕЗУЛЬТАТ ОТ LLM (ТИПИЗИРОВАННЫЙ) ---")
        print(f"Тип объекта: {type(decision)}")
        print(f"Мысли агента: {decision.thoughts}")
        print(f"Действие: {decision.action}")
        print(f"Сумма: {decision.amount}")
        
    except Exception as e:
        print(f"Ошибка при генерации: {e}")

if __name__ == "__main__":
    main()
