import sys
from src.utils.config import load_config
from src.llm.factory import get_llm_backend
from src.core.engine import SimulationEngine

def main():
    cfg = load_config()
    
    llm = get_llm_backend(cfg)
    
    engine = SimulationEngine(llm, cfg)
    engine.run(steps=3)

if __name__ == "__main__":
    main()
