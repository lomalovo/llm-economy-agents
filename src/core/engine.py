import asyncio
from src.core.state import WorldState
from src.agents.impl import HouseholdAgent, FirmAgent
from src.economics.mechanism import MarketMechanism
from src.llm.backend.base import BaseLLMBackend
from src.core.logger import SimulationLogger
from src.core.builder import AgentBuilder
from src.llm.prompt_manager import get_prompt_manager

REFLECTION_SYSTEM_PROMPT = (
    "You are an economic agent in a macro simulation. "
    "Reflect on the period that just passed and draw brief conclusions from the numbers. "
    "Respond in plain text, no JSON, 2–4 sentences."
)


class SimulationEngine:
    def __init__(self, llm: BaseLLMBackend, config: dict):
        self.llm = llm
        self.cfg = config
        self.state = WorldState()

        # Override WorldState defaults from config (calibration)
        for key, val in config.get("initial_state", {}).items():
            if hasattr(self.state, key):
                setattr(self.state, key, val)

        market_cfg    = config.get("market", {})
        self.market   = MarketMechanism(
            goods_clearing_mode=market_cfg.get("goods_clearing_mode", "average"),
            matching_efficiency=market_cfg.get("matching_efficiency", 1.0),
            matching_elasticity=market_cfg.get("matching_elasticity", 0.5),
            separation_rate=market_cfg.get("separation_rate", 0.0),
        )
        self._phi      = market_cfg.get("wage_adjustment_speed", 0.15)
        self._wage_max_up   = market_cfg.get("wage_max_increase", 0.10)
        self._wage_max_down = market_cfg.get("wage_max_decrease", 0.05)
        # Price stickiness: 1.0 = no filter (LLM decisions flow directly),
        # <1.0 = Calvo-style partial adjustment. Kept configurable so it can be
        # part of the MSM calibration vector θ rather than a hidden artifact.
        self._price_adjustment_speed = market_cfg.get("price_adjustment_speed", 1.0)

        sim_cfg = config.get("simulation", {})
        self._reflection_every = sim_cfg.get("reflection_every", 3)

        # gov_cfg and cb_cfg are stored by reference so tax_change events
        # that mutate self.cfg["government"]["tax_brackets"] are reflected here.
        self._gov_cfg = config.get("government", {})
        self._cb_cfg  = config.get("central_bank", {})

        run_name = config.get("experiment", {}).get("name", "sim_run")
        self.logger = SimulationLogger(run_name)

        # Optionally calibrate WorldState from FRED data
        if config.get("experiment", {}).get("use_fred_init", False):
            from src.utils.fred_init import apply_to_world_state
            apply_to_world_state(self.state)

        self.households = []
        self.firms = []

    def setup(self):
        self.households, self.firms = AgentBuilder.build_from_config(self.llm, self.cfg)
        print(f"Setup complete: {len(self.households)} HHs, {len(self.firms)} Firms.")

    def _process_events(self):
        for event in self.cfg.get("events", []):
            if event["step"] != self.state.step:
                continue
            print(f"EVENT TRIGGERED: {event['description']}")

            if event["type"] == "cash_injection":
                amount      = event["amount"]
                target_type = event["target"]
                targets     = self.households if target_type == "household" else (
                              self.firms      if target_type == "firm"      else [])
                for agent in targets:
                    agent.money += amount
                    agent.pending_event_notifications.append(
                        f"NOTICE: You received a one-time government transfer of "
                        f"{amount:.0f} this period. This is new money added to your balance."
                    )
                # Notify firms so they can respond to demand surge immediately
                if target_type == "household":
                    n_hh = len(self.households)
                    total_injection = amount * n_hh
                    for f in self.firms:
                        f.pending_event_notifications.append(
                            f"MARKET SIGNAL: The government has distributed {amount:.0f} to each "
                            f"of {n_hh} households (total injection: {total_injection:.0f}). "
                            f"Consumer purchasing power has increased significantly this period. "
                            f"Expect higher demand — consider adjusting your hiring and production accordingly."
                        )
                print(f"Cash injection: {amount} to each {target_type}")

            elif event["type"] == "supply_shock":
                multiplier = event.get("multiplier", 1.0)
                for f in self.firms:
                    f.inventory = max(0.0, round(f.inventory * multiplier, 4))
                print(f"Supply shock: inventory × {multiplier}")

            elif event["type"] == "tax_change":
                new_brackets = event.get("tax_brackets")
                if new_brackets and self._gov_cfg.get("enabled", False):
                    self._gov_cfg["tax_brackets"] = new_brackets
                    print(f"Tax brackets updated: {new_brackets}")
                else:
                    print("tax_change: government disabled or no tax_brackets provided")

            elif event["type"] == "productivity_shock":
                multiplier = event.get("multiplier", 1.0)
                for f in self.firms:
                    f.productivity = round(f.productivity * multiplier, 4)
                    f.pending_event_notifications.append(
                        f"PRODUCTION UPDATE: Your workforce efficiency has increased by "
                        f"{(multiplier - 1) * 100:.0f}% this period due to a technology improvement. "
                        f"Each worker now produces {f.productivity:.2f}x more output. "
                        f"Your effective cost per unit has decreased proportionally."
                    )
                first = self.firms[0].productivity if self.firms else "—"
                print(f"Productivity shock: × {multiplier} (firms now at {first})")

            else:
                print(f"Unknown event type: {event['type']!r}")

    @staticmethod
    def _calculate_tax(income: float, sorted_brackets: list) -> float:
        """Прогрессивный налог; принимает уже отсортированные по threshold брекеты."""
        if income <= 0:
            return 0.0
        tax = 0.0
        for i, bracket in enumerate(sorted_brackets):
            lower = bracket["threshold"]
            rate  = bracket["rate"]
            upper = sorted_brackets[i + 1]["threshold"] if i + 1 < len(sorted_brackets) else float("inf")
            if income <= lower:
                break
            tax += (min(income, upper) - lower) * rate
        return round(tax, 4)

    def _collect_taxes_and_redistribute(self):
        if not self._gov_cfg.get("enabled", False):
            return
        brackets = self._gov_cfg.get("tax_brackets", [])
        if not brackets:
            return

        # Sort once per step (brackets may change via tax_change event)
        sorted_brackets = sorted(brackets, key=lambda b: b["threshold"])

        total_tax = 0.0
        for h in self.households:
            income  = h.last_worked * self.state.avg_wage
            tax     = self._calculate_tax(income, sorted_brackets)
            tax     = min(tax, h.money)
            h.money -= tax
            h.last_tax_paid = tax
            total_tax += tax

        n = len(self.households)
        redistribution_per_hh = round(total_tax / n, 4) if n > 0 else 0.0
        for h in self.households:
            h.money              += redistribution_per_hh
            h.last_redistribution = redistribution_per_hh

        self.state.last_tax_collected  = round(total_tax, 2)
        self.state.last_redistribution = round(redistribution_per_hh, 2)
        print(f"Gov: tax_collected={total_tax:.2f}, redistribution={redistribution_per_hh:.2f}/HH")

    def _update_interest_rate(self):
        """Правило Тейлора: ЦБ устанавливает ставку на основе инфляции и безработицы."""
        if not self._cb_cfg.get("enabled", False):
            return

        r_neutral   = self._cb_cfg.get("neutral_rate", 0.05)
        target_inf  = self._cb_cfg.get("target_inflation", 0.02)
        target_unem = self._cb_cfg.get("target_unemployment", 0.05)
        alpha       = self._cb_cfg.get("inflation_sensitivity", 1.5)
        beta        = self._cb_cfg.get("unemployment_sensitivity", 0.5)
        min_rate    = self._cb_cfg.get("min_rate", 0.0)
        max_rate    = self._cb_cfg.get("max_rate", 0.25)

        inflation = (
            (self.state.avg_price - self.state.prev_avg_price) / self.state.prev_avg_price
            if self.state.prev_avg_price > 0 else 0.0
        )
        r = r_neutral + alpha * (inflation - target_inf) + beta * (target_unem - self.state.unemployment_rate)
        old_rate = self.state.interest_rate
        self.state.interest_rate = round(max(min_rate, min(max_rate, r)), 4)
        print(f"CB: inflation={inflation:+.2%}, rate: {old_rate:.2%} → {self.state.interest_rate:.2%}")

    def _apply_interest_on_savings(self):
        """Начисляет процентный доход на сбережения домохозяйств (из решения прошлого шага)."""
        if not self._cb_cfg.get("enabled", False):
            return
        rate = self.state.interest_rate
        for h in self.households:
            interest           = round(h.last_savings * rate, 4)
            h.money           += interest
            h.last_interest_earned = interest
            # Фиксируем сбережения текущего шага для следующего начисления
            h.last_savings     = h.current_decision.savings_amount if h.current_decision else 0.0

    async def _run_reflections(self, wage_trend: str):
        """Параллельная рефлексия всех агентов."""
        prompter = get_prompt_manager()

        world_ctx = {
            "step":             self.state.step,
            "avg_price":        round(self.state.avg_price, 2),
            "avg_wage":         round(self.state.avg_wage, 2),
            "unemployment_pct": round(self.state.unemployment_rate * 100, 1),
            "wage_trend":       wage_trend,
        }

        tasks        = []
        agents_order = []

        # Precompute household totals for O(1) neighbor averages
        n_hh = len(self.households)
        if n_hh > 1:
            hh_total_money  = sum(x.money       for x in self.households)
            hh_total_spent  = sum(x.last_spent  for x in self.households)
            hh_total_worked = sum(x.last_worked for x in self.households)

        for h in self.households:
            n = n_hh - 1
            if n > 0:
                neighbor_ctx = {
                    "neighbor_avg_money":  round((hh_total_money  - h.money)       / n, 2),
                    "neighbor_avg_spent":  round((hh_total_spent  - h.last_spent)  / n, 2),
                    "neighbor_avg_worked": round((hh_total_worked - h.last_worked) / n, 2),
                }
            else:
                neighbor_ctx = {"neighbor_avg_money": 0, "neighbor_avg_spent": 0, "neighbor_avg_worked": 0}

            ctx    = {**world_ctx, **neighbor_ctx, "history": h.history,
                      "agent_id": h.id, "money": round(h.money, 2), **h.attributes}
            prompt = prompter.render("reflection_household.j2", **ctx)
            tasks.append(self.llm.generate(system_prompt=REFLECTION_SYSTEM_PROMPT,
                                           user_prompt=prompt, schema=None))
            agents_order.append(h)

        # Precompute firm totals for O(1) neighbor averages
        n_f = len(self.firms)
        if n_f > 1:
            f_total_money     = sum(x.money     for x in self.firms)
            f_total_inventory = sum(x.inventory for x in self.firms)
            f_total_sold      = sum(x.goods_sold for x in self.firms)
            f_total_price     = sum(
                x.current_decision.price_setting for x in self.firms if x.current_decision
            )
            f_with_decision = sum(1 for x in self.firms if x.current_decision)

        for f in self.firms:
            n = n_f - 1
            if n > 0:
                f_price = f.current_decision.price_setting if f.current_decision else 0.0
                others_with_dec = f_with_decision - (1 if f.current_decision else 0)
                neighbor_avg_price = (
                    round((f_total_price - f_price) / others_with_dec, 2)
                    if others_with_dec > 0 else 0
                )
                neighbor_ctx = {
                    "neighbor_avg_money":     round((f_total_money     - f.money)     / n, 2),
                    "neighbor_avg_inventory": round((f_total_inventory - f.inventory) / n, 2),
                    "neighbor_avg_sold":      round((f_total_sold      - f.goods_sold) / n, 2),
                    "neighbor_avg_price":     neighbor_avg_price,
                    "goods_sold":             round(f.goods_sold, 2),
                    "inventory":              round(f.inventory, 2),
                }
            else:
                neighbor_ctx = {
                    "neighbor_avg_money": 0, "neighbor_avg_inventory": 0,
                    "neighbor_avg_sold": 0, "neighbor_avg_price": 0,
                    "goods_sold": round(f.goods_sold, 2), "inventory": round(f.inventory, 2),
                }

            ctx    = {**world_ctx, **neighbor_ctx, "history": f.history,
                      "agent_id": f.id, **f.attributes}
            prompt = prompter.render("reflection_firm.j2", **ctx)
            tasks.append(self.llm.generate(system_prompt=REFLECTION_SYSTEM_PROMPT,
                                           user_prompt=prompt, schema=None))
            agents_order.append(f)

        print(f"  Reflection phase: {len(tasks)} agents thinking...")
        results = await asyncio.gather(*tasks)

        for agent, text in zip(agents_order, results):
            if isinstance(text, str) and text.strip():
                agent.update_reflection(text, self.state.step)

    async def step(self):
        self.state.step += 1
        print(f"\n--- STEP {self.state.step} START ---")

        prev_wage = self.state.avg_wage

        self._process_events()

        # Pre-decision market context should reflect the most recent realized state,
        # not stale planned demand from the previous decision vector.
        total_last_spending = sum(h.last_spent for h in self.households)
        total_last_bought = sum(h.last_bought for h in self.households)
        total_supply_qty = sum(f.inventory for f in self.firms)
        sell_ratio = round(total_last_bought / total_supply_qty, 2) if total_supply_qty > 0 else "N/A"

        # Compute inflation from the step that just finished (avg vs prev)
        inflation = (
            (self.state.avg_price - self.state.prev_avg_price) / self.state.prev_avg_price
            if self.state.prev_avg_price > 0 else 0.0
        )

        market_info = {
            "step":                  self.state.step,
            "wage":                  self.state.avg_wage,
            "price":                 self.state.avg_price,
            "last_demand":           round(total_last_bought, 1),
            "total_consumer_demand": round(total_last_spending, 1),
            "sell_ratio":            sell_ratio,
            "unemployment_rate":     round(self.state.unemployment_rate, 4),
            "vacancy_rate":          round(self.state.vacancy_rate, 4),
            "inflation_rate":        round(inflation, 4),
            "last_redistribution":   round(self.state.last_redistribution, 2),
            "last_tax_collected":    round(self.state.last_tax_collected, 2),
            "interest_rate":         round(self.state.interest_rate, 4),
        }

        print("Agents are thinking (parallel)...")
        await asyncio.gather(*(agent.make_decision(market_info) for agent in self.households + self.firms))

        self.market.clear_labor_market(self.firms, self.households, self.state.avg_wage)

        labor_force = self.market.last_labor_supply
        unemp_units = self.market.last_unemployment_units
        vac_units   = self.market.last_vacancies_units
        labor_demand = self.market.last_labor_demand

        # Rates normalized by labor force (participation pool) — Beveridge convention.
        if labor_force > 0:
            self.state.unemployment_rate = round(unemp_units / labor_force, 4)
            self.state.vacancy_rate      = round(vac_units   / labor_force, 4)
        else:
            self.state.unemployment_rate = 0.0
            self.state.vacancy_rate      = 0.0

        # Wage adjustment uses the same (declared) imbalance signal as before
        if labor_force > 0:
            imbalance  = (labor_demand - labor_force) / labor_force
            adjustment = max(-self._wage_max_down, min(self._wage_max_up, imbalance * self._phi))
            old_wage   = self.state.avg_wage
            self.state.avg_wage = max(1.0, self.state.avg_wage * (1 + adjustment))
            if abs(adjustment) > 0.005:
                print(f"Wage: {old_wage:.2f} → {self.state.avg_wage:.2f} (imbalance={imbalance:+.2%}, adj={adjustment:+.2%})")

        self._collect_taxes_and_redistribute()

        sold = self.market.clear_goods_market(self.firms, self.households, self.state.avg_price)

        prices = [f.current_decision.price_setting for f in self.firms if f.current_decision]
        if prices:
            desired_price = sum(prices) / len(prices)
            speed = self._price_adjustment_speed
            if self.state.step == 1:
                new_price = desired_price
                self.state.prev_avg_price = new_price
            elif speed >= 1.0:
                # No filter — LLM price decisions flow directly to the market
                new_price = desired_price
                self.state.prev_avg_price = self.state.avg_price
            else:
                new_price = self.state.avg_price + speed * (desired_price - self.state.avg_price)
                self.state.prev_avg_price = self.state.avg_price
            self.state.avg_price = round(new_price, 4)

        self._update_interest_rate()
        self._apply_interest_on_savings()

        print(f"Stats: Unemp={self.state.unemployment_rate:.0%}, Sold={sold:.1f}, Price={self.state.avg_price:.2f}, Wage={self.state.avg_wage:.2f}, Rate={self.state.interest_rate:.2%}")

        for h in self.households:
            h.update_history(self.state.step, self.state.avg_wage)
        for f in self.firms:
            f.update_history(self.state.step)

        if self.state.step % self._reflection_every == 0:
            wage_change = (self.state.avg_wage - prev_wage) / prev_wage * 100 if prev_wage > 0 else 0
            if wage_change > 0.5:
                wage_trend = f"растёт (+{wage_change:.1f}%)"
            elif wage_change < -0.5:
                wage_trend = f"падает ({wage_change:.1f}%)"
            else:
                wage_trend = "стабильна"
            print(f"\nREFLECTION PHASE (step={self.state.step}, every={self._reflection_every})...")
            await self._run_reflections(wage_trend)

        self.logger.log_step(self.state.step, self.state, self.firms, self.households)

    async def run(self, steps=3):
        self.setup()
        try:
            for _ in range(steps):
                await self.step()
        finally:
            self.logger.save()
