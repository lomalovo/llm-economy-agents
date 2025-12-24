from pathlib import Path
from jinja2 import Environment, FileSystemLoader

class PromptManager:
    def __init__(self, template_dir="templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))
    
    def render(self, template_name: str, **kwargs) -> str:
        """
        template_name: 'household.j2'
        kwargs: переменные для вставки (money, price, greed...)
        """
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

_instance = None

def get_prompt_manager():
    global _instance
    if _instance is None:
        _instance = PromptManager()
    return _instance
