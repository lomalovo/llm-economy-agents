from src.agents.impl import HouseholdAgent, FirmAgent

class AgentBuilder:
    @staticmethod
    def build_from_config(llm, config: dict):
        households = []
        firms = []
        
        agents_cfg = config.get("agents", {})
        
        # Глобальные параметры симуляции, которые инжектируются в каждого агента
        sim_cfg = config.get("simulation", {})
        global_agent_params = {
            "history_window":    sim_cfg.get("history_window", 5),
            "reflection_window": sim_cfg.get("reflection_window", 3),
        }

        for group_name, group_data in agents_cfg.items():
            agent_type = group_data.get("type")
            count = group_data.get("count", 1)
            # Мержим: глобальные параметры + параметры группы (группа имеет приоритет)
            params = {**global_agent_params, **group_data.get("params", {})}
            
            print(f"Building group '{group_name}' ({count} agents)...")
            
            for i in range(count):
                agent_id = f"{group_name}_{i+1}"

                # If a list of bios is provided, assign one per agent
                params_i = dict(params)
                if "bios" in params_i:
                    bios_list = params_i.pop("bios")
                    params_i["bio"] = bios_list[i % len(bios_list)]

                if agent_type == "household":
                    agent = HouseholdAgent(agent_id, llm, **params_i)
                    households.append(agent)
                    
                elif agent_type == "firm":
                    agent = FirmAgent(agent_id, llm, **params_i)
                    firms.append(agent)
                
                else:
                    print(f"Unknown agent type: {agent_type}")
                    
        return households, firms
