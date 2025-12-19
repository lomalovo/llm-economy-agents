import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Загружает конфигурацию из yaml файла."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
