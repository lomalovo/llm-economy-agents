import re
from json_repair import repair_json

def extract_json_from_text(text: str) -> str:
    """
    Пытается найти и починить JSON внутри ответа LLM.
    """
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(1)
    
    try:
        fixed_json = repair_json(text, return_objects=False, skip_json_loads=True)
        return fixed_json
    except Exception:
        return text
