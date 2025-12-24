from src.agents.impl import HouseholdAgent, FirmAgent

class AgentBuilder:
    @staticmethod
    def build_from_config(llm, config: dict):
        households = []
        firms = []
        
        agents_cfg = config.get("agents", {})
        
        for group_name, group_data in agents_cfg.items():
            agent_type = group_data.get("type")
            count = group_data.get("count", 1)
            params = group_data.get("params", {})
            
            print(f"Building group '{group_name}' ({count} agents)...")
            
            for i in range(count):
                # Формируем уникальный ID: rich_savers_1, ich_savers_2
                agent_id = f"{group_name}_{i+1}"
                
                if agent_type == "household":
                    agent = HouseholdAgent(agent_id, llm, **params)
                    households.append(agent)
                    
                elif agent_type == "firm":
                    agent = FirmAgent(agent_id, llm, **params)
                    firms.append(agent)
                
                else:
                    print(f"Unknown agent type: {agent_type}")
                    
        return households, firms
