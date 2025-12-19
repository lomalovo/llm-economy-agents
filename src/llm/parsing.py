import json
import re

def extract_json_from_text(text: str) -> str:
    """
    Пытается найти JSON-блок внутри текста от LLM.
    Убирает маркдаун обертки ```json ... ```.
    """
    text = text.strip()
    
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        return json_str
    
    return text
