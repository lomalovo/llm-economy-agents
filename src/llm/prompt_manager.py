from jinja2 import Environment, FileSystemLoader

# Shared system prompt for all economic agents. Kept intentionally light — the
# heavy lifting of persona and behaviour is in the Jinja templates / bios.
SYSTEM_PROMPT_ECON = (
    "You are an economic agent in a macro simulation. "
    "Act consistently with the personality and background described in your prompt. "
    "Your decisions affect prices, wages, and the wellbeing of all agents in the economy."
)


class PromptManager:
    def __init__(self, template_dir: str = "templates"):
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def render(self, template_name: str, **kwargs) -> str:
        """Render a Jinja template. `template_name` is e.g. 'household.j2'."""
        template = self.env.get_template(template_name)
        return template.render(**kwargs)


_instance: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _instance
    if _instance is None:
        _instance = PromptManager()
    return _instance
