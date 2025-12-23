import sys
import asyncio
from src.utils.config import load_config
from src.llm.factory import get_llm_backend
from src.core.engine import SimulationEngine

async def async_main():
    cfg = load_config()
    
    llm = get_llm_backend(cfg)
    
    engine = SimulationEngine(llm, cfg)
    
    await engine.run(steps=5)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
