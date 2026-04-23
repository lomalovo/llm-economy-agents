import random
from typing import List, Tuple
from src.agents.impl import HouseholdAgent, FirmAgent
from src.schemas.economics import FirmDecision, HouseholdDecision

VALID_CLEARING_MODES = ("average", "queue", "weighted")


class MarketMechanism:
    def __init__(
        self,
        goods_clearing_mode: str = "average",
        matching_efficiency: float = 1.0,
        matching_elasticity: float = 0.5,
        separation_rate: float = 0.0,
    ):
        if goods_clearing_mode not in VALID_CLEARING_MODES:
            raise ValueError(
                f"Unknown goods_clearing_mode '{goods_clearing_mode}'. "
                f"Valid options: {VALID_CLEARING_MODES}"
            )
        self.goods_clearing_mode = goods_clearing_mode

        # Mortensen–Pissarides Cobb–Douglas matching: H = A · U^α · V^(1-α)
        # A = 1.0 collapses to H = min(U, V) (legacy frictionless behavior).
        # A < 1.0 introduces matching friction — simultaneously positive U and V
        # (required for a non-degenerate Beveridge curve).
        self.matching_efficiency = matching_efficiency
        self.matching_elasticity = matching_elasticity

        # Separation rate: fraction of last-period's filled matches that break
        # at the start of each period. Creates independent vacancy inflow.
        self.separation_rate = separation_rate

        self.last_labor_supply: float = 0.0
        self.last_labor_demand: float = 0.0
        self.last_matches: float = 0.0
        self.last_unemployment_units: float = 0.0
        self.last_vacancies_units: float = 0.0

    # ------------------------------------------------------------------ #
    #  РЫНОК ТРУДА                                                         #
    # ------------------------------------------------------------------ #

    def clear_labor_market(
        self,
        firms: List[FirmAgent],
        households: List[HouseholdAgent],
        wage: float,
    ):
        """Labor market clearing with Cobb–Douglas matching function.

        H = A · U^α · V^(1-α), capped at min(U, V). With A=1 this collapses to
        min(U, V) (frictionless). With A<1, matches < min(U, V), so both
        unmatched unemployment and unfilled vacancies coexist — standard
        Mortensen–Pissarides friction.

        Separation rate s: if last-period matches were M_prev, at start of this
        period s·M_prev workers return to the seeker pool, generating vacancy
        inflow independent of current-period decisions.
        """

        # 1. Aggregate supply and demand
        total_supply = sum(
            h.current_decision.labor_supply
            for h in households
            if h.current_decision is not None
        )
        for h in households:
            if h.current_decision is None:
                h.last_worked = 0.0

        total_demand = sum(
            f.current_decision.labor_demand
            for f in firms
            if f.current_decision is not None
        )

        # 2. Apply separation: s·M_prev workers returned to the seeker pool.
        # This inflates effective unemployment and vacancies by the same amount
        # (these jobs need re-matching), which is the canonical Beveridge inflow.
        separation_inflow = self.separation_rate * self.last_matches
        effective_supply = total_supply + separation_inflow
        effective_demand = total_demand + separation_inflow

        # 3. Cobb–Douglas matching
        if effective_supply > 0 and effective_demand > 0:
            A = self.matching_efficiency
            a = self.matching_elasticity
            raw_matches = A * (effective_supply ** a) * (effective_demand ** (1 - a))
            matches = min(raw_matches, effective_supply, effective_demand)
        else:
            matches = 0.0

        unemployment_units = max(0.0, effective_supply - matches)
        vacancies_units    = max(0.0, effective_demand - matches)

        self.last_labor_supply         = total_supply
        self.last_labor_demand         = total_demand
        self.last_matches              = matches
        self.last_unemployment_units   = unemployment_units
        self.last_vacancies_units      = vacancies_units

        # 4. Rationing ratios over *declared* (non-separation) quantities so
        # firms and households see the impact on what they asked for.
        labor_ratio      = min(1.0, matches / effective_demand) if effective_demand > 0 else 0.0
        employment_ratio = min(1.0, matches / effective_supply) if effective_supply > 0 else 0.0

        # 5. Firms pay wages for their hired share
        for f in firms:
            if f.current_decision is None:
                f.labor_hired = 0.0
                continue
            hired = f.current_decision.labor_demand * labor_ratio
            cost  = hired * wage
            if f.money >= cost:
                f.money      -= cost
                f.labor_hired = hired
            else:
                f.labor_hired = 0.0

        # 6. Households receive wages proportional to their labor supply
        for h in households:
            if h.current_decision is None:
                continue
            h.last_worked = h.current_decision.labor_supply * employment_ratio
            h.money      += h.last_worked * wage

        return matches, employment_ratio

    # ------------------------------------------------------------------ #
    #  ТОВАРНЫЙ РЫНОК — диспетчер                                         #
    # ------------------------------------------------------------------ #

    def _produce(self, firms: List[FirmAgent]) -> None:
        """Производство: Y = min(L × productivity, production_target). Сбрасывает goods_sold."""
        for f in firms:
            capacity = f.labor_hired * f.productivity
            target   = f.current_decision.production_target if f.current_decision else capacity
            f.inventory  += min(capacity, target)
            f.goods_sold  = 0.0

    def clear_goods_market(
        self,
        firms: List[FirmAgent],
        households: List[HouseholdAgent],
        avg_price: float,
    ) -> float:
        """Производство + клиринг товарного рынка в выбранном режиме."""
        self._produce(firms)
        if self.goods_clearing_mode == "average":
            return self._clear_goods_average(firms, households, avg_price)
        elif self.goods_clearing_mode == "queue":
            return self._clear_goods_queue(firms, households)
        elif self.goods_clearing_mode == "weighted":
            return self._clear_goods_weighted(firms, households)
        return 0.0  # недостижимо, но нужно для type checker

    # ------------------------------------------------------------------ #
    #  Режим 1: AVERAGE                                                    #
    # ------------------------------------------------------------------ #

    def _clear_goods_average(
        self,
        firms: List[FirmAgent],
        households: List[HouseholdAgent],
        avg_price: float,
    ) -> float:
        """Все фирмы торгуют по единой средней цене; доход делится пропорционально инвентарю."""
        total_money = sum(
            h.current_decision.consumption_budget
            for h in households
            if h.current_decision is not None
        )

        total_available = sum(f.inventory for f in firms)
        max_can_buy     = total_money / avg_price if avg_price > 0 else 0.0
        actual_sold     = min(total_available, max_can_buy)

        revenue = actual_sold * avg_price
        if total_available > 0:
            for f in firms:
                share        = f.inventory / total_available
                sold_here    = actual_sold * share
                f.goods_sold  = sold_here
                f.money      += revenue * share
                f.inventory  -= sold_here

        if max_can_buy > 0:
            fill_rate = actual_sold / max_can_buy
            for h in households:
                if h.current_decision is None:
                    continue
                spent        = h.current_decision.consumption_budget * fill_rate
                h.money     -= spent
                h.last_spent  = spent
                h.last_bought = spent / avg_price if avg_price > 0 else 0.0

        return actual_sold

    # ------------------------------------------------------------------ #
    #  Режим 2: QUEUE — случайная очередь, покупают от дешёвой фирмы      #
    # ------------------------------------------------------------------ #

    def _clear_goods_queue(
        self,
        firms: List[FirmAgent],
        households: List[HouseholdAgent],
    ) -> float:
        """Покупатели в случайном порядке идут от самой дешёвой фирмы к дорогой."""
        firm_pairs: List[Tuple[FirmAgent, FirmDecision]] = [
            (f, f.current_decision)
            for f in firms
            if f.current_decision is not None
        ]
        firm_pairs.sort(key=lambda pair: pair[1].price_setting)

        hh_pairs: List[Tuple[HouseholdAgent, HouseholdDecision]] = [
            (h, h.current_decision)
            for h in households
            if h.current_decision is not None
        ]
        random.shuffle(hh_pairs)

        for h, _ in hh_pairs:
            h.last_spent  = 0.0
            h.last_bought = 0.0

        total_sold = 0.0
        for h, h_dec in hh_pairs:
            remaining = h_dec.consumption_budget
            for firm, f_dec in firm_pairs:
                if remaining <= 1e-9 or firm.inventory <= 1e-9:
                    continue
                price = f_dec.price_setting
                qty   = min(remaining / price, firm.inventory)
                spent = qty * price

                firm.money      += spent
                firm.inventory  -= qty
                firm.goods_sold += qty

                h.money       -= spent
                h.last_spent  += spent
                h.last_bought += qty
                remaining     -= spent
                total_sold    += qty

        return total_sold

    # ------------------------------------------------------------------ #
    #  Режим 3: WEIGHTED — спрос обратно пропорционально цене              #
    # ------------------------------------------------------------------ #

    def _clear_goods_weighted(
        self,
        firms: List[FirmAgent],
        households: List[HouseholdAgent],
    ) -> float:
        """Спрос между фирмами пропорционально 1/price; излишек при дефиците пропадает."""
        firm_pairs: List[Tuple[FirmAgent, FirmDecision]] = [
            (f, f.current_decision)
            for f in firms
            if f.current_decision is not None
        ]
        if not firm_pairs:
            return 0.0

        inv_prices  = [1.0 / f_dec.price_setting for _, f_dec in firm_pairs]
        total_inv_p = sum(inv_prices)
        weights     = [w / total_inv_p for w in inv_prices]

        hh_pairs: List[Tuple[HouseholdAgent, HouseholdDecision]] = [
            (h, h.current_decision)
            for h in households
            if h.current_decision is not None
        ]
        total_budget = sum(h_dec.consumption_budget for _, h_dec in hh_pairs)

        firm_fill: dict[str, float] = {}
        total_sold = 0.0

        for (f, f_dec), w in zip(firm_pairs, weights):
            money_to_firm = total_budget * w
            qty_demanded  = money_to_firm / f_dec.price_setting
            actually_sold = min(qty_demanded, f.inventory)
            fill_rate     = actually_sold / qty_demanded if qty_demanded > 1e-9 else 0.0

            f.goods_sold    = actually_sold
            f.money        += actually_sold * f_dec.price_setting
            f.inventory    -= actually_sold
            total_sold     += actually_sold
            firm_fill[f.id] = fill_rate

        for h, h_dec in hh_pairs:
            budget       = h_dec.consumption_budget
            total_spent  = 0.0
            total_bought = 0.0

            for (f, f_dec), w in zip(firm_pairs, weights):
                intended_spend = budget * w
                fill_rate      = firm_fill[f.id]
                actual_spend   = intended_spend * fill_rate
                qty_bought     = actual_spend / f_dec.price_setting if f_dec.price_setting > 1e-9 else 0.0
                total_spent  += actual_spend
                total_bought += qty_bought

            h.money      -= total_spent
            h.last_spent  = total_spent
            h.last_bought = total_bought

        return total_sold
