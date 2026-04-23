"""
Unit tests for MarketMechanism clearing logic.

Run:  pytest tests/test_clearing.py -v
"""
import pytest
from src.agents.impl import HouseholdAgent, FirmAgent
from src.schemas.economics import HouseholdDecision, FirmDecision
from src.economics.mechanism import MarketMechanism
from src.llm.backend.dummy import DummyBackend

_DUMMY = DummyBackend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_household(agent_id, money, labor_supply, budget, savings=0.0):
    h = HouseholdAgent(agent_id, _DUMMY, initial_money=money)
    h.current_decision = HouseholdDecision(
        reasoning="test",
        labor_supply=labor_supply,
        consumption_budget=budget,
        savings_amount=savings,
    )
    return h


def make_firm(agent_id, capital, labor_demand, price, production_target,
              labor_hired=0.0, inventory=0.0, productivity=1.0):
    f = FirmAgent(agent_id, _DUMMY, initial_capital=capital, productivity=productivity)
    f.current_decision = FirmDecision(
        reasoning="test",
        labor_demand=labor_demand,
        price_setting=price,
        production_target=production_target,
    )
    f.labor_hired = labor_hired
    f.inventory   = inventory
    return f


# ---------------------------------------------------------------------------
# Labor market
# ---------------------------------------------------------------------------

class TestLaborMarket:
    def setup_method(self):
        self.market = MarketMechanism()

    def test_balanced_market(self):
        """Equal supply and demand → all hired, full employment."""
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(2)]
        firms = [make_firm("f1", 1000, 2.0, 10, 10)]

        actual_labor, emp_ratio = self.market.clear_labor_market(firms, hh, wage=10.0)

        assert actual_labor == pytest.approx(2.0)
        assert emp_ratio    == pytest.approx(1.0)
        assert firms[0].labor_hired == pytest.approx(2.0)

    def test_excess_demand(self):
        """Firms want more workers than available → employment ratio < 1."""
        hh    = [make_household("h1", 100, 1.0, 20)]
        firms = [make_firm("f1", 1000, 4.0, 10, 10)]

        actual_labor, emp_ratio = self.market.clear_labor_market(firms, hh, wage=5.0)

        assert actual_labor == pytest.approx(1.0)
        assert emp_ratio    == pytest.approx(1.0)
        assert firms[0].labor_hired == pytest.approx(1.0)

    def test_excess_supply(self):
        """More workers than firms want → partial employment."""
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(4)]
        firms = [make_firm("f1", 1000, 2.0, 10, 10)]

        actual_labor, emp_ratio = self.market.clear_labor_market(firms, hh, wage=5.0)

        assert actual_labor == pytest.approx(2.0)
        assert emp_ratio    == pytest.approx(0.5)
        assert firms[0].labor_hired == pytest.approx(2.0)

    def test_firm_cannot_afford_wage(self):
        """Firm with zero capital cannot pay wages → hires 0."""
        hh    = [make_household("h1", 100, 1.0, 20)]
        firms = [make_firm("f1", capital=0.0, labor_demand=1.0, price=10, production_target=5)]

        self.market.clear_labor_market(firms, hh, wage=100.0)

        assert firms[0].labor_hired == pytest.approx(0.0)

    def test_household_income(self):
        """Household receives wage * employment_ratio."""
        hh    = [make_household("h1", 100, 1.0, 20)]
        firms = [make_firm("f1", 1000, 1.0, 10, 5)]

        self.market.clear_labor_market(firms, hh, wage=10.0)

        assert hh[0].money == pytest.approx(110.0)


class TestMatchingFunction:
    """Mortensen-Pissarides matching: H = A·U^α·V^(1-α). A<1 creates Beveridge friction."""

    def test_matching_efficiency_one_equals_frictionless(self):
        """A=1.0, balanced market → matches equal to min(U,V)."""
        m = MarketMechanism(matching_efficiency=1.0)
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(3)]
        firms = [make_firm("f1", 1000, 3.0, 10, 10)]
        matches, _ = m.clear_labor_market(firms, hh, wage=5.0)
        # At V=U=3 with A=1: sqrt(9)=3 = min, so matches=3, no friction
        assert matches == pytest.approx(3.0)
        assert m.last_unemployment_units == pytest.approx(0.0)
        assert m.last_vacancies_units    == pytest.approx(0.0)

    def test_friction_creates_simultaneous_u_and_v(self):
        """A=0.6, balanced market → matches=1.8, both U and V strictly positive."""
        m = MarketMechanism(matching_efficiency=0.6)
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(3)]
        firms = [make_firm("f1", 1000, 3.0, 10, 10)]
        matches, _ = m.clear_labor_market(firms, hh, wage=5.0)
        # H = 0.6 · sqrt(3·3) = 1.8
        assert matches == pytest.approx(1.8)
        assert m.last_unemployment_units == pytest.approx(1.2)
        assert m.last_vacancies_units    == pytest.approx(1.2)

    def test_friction_unbalanced_market(self):
        """A=0.5, V=4, U=9: matches=0.5·sqrt(36)=3, u=6, v=1."""
        m = MarketMechanism(matching_efficiency=0.5)
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(9)]
        firms = [make_firm("f1", 1000, 4.0, 10, 10)]
        matches, _ = m.clear_labor_market(firms, hh, wage=5.0)
        assert matches == pytest.approx(3.0)
        assert m.last_unemployment_units == pytest.approx(6.0)
        assert m.last_vacancies_units    == pytest.approx(1.0)

    def test_separation_adds_inflow_to_second_period(self):
        """separation_rate=0.2 → 20% of last matches re-enter seeker pool next period."""
        m = MarketMechanism(matching_efficiency=1.0, separation_rate=0.2)
        hh    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(3)]
        firms = [make_firm("f1", 1000, 3.0, 10, 10)]
        m.clear_labor_market(firms, hh, wage=5.0)
        # After period 1: matches=3, no friction
        assert m.last_matches == pytest.approx(3.0)

        # Period 2: separation returns 0.6 units to seeker pool, but new supply still 3
        # effective_supply = 3 + 0.6 = 3.6, effective_demand = 3 + 0.6 = 3.6
        # H = 1.0 · sqrt(3.6·3.6) = 3.6 → matches = 3.6 (capped at min = 3.6)
        hh2    = [make_household(f"h{i}", 100, 1.0, 20) for i in range(3)]
        firms2 = [make_firm("f1", 1000, 3.0, 10, 10)]
        m.clear_labor_market(firms2, hh2, wage=5.0)
        assert m.last_matches == pytest.approx(3.6)


# ---------------------------------------------------------------------------
# Goods market — average mode
# ---------------------------------------------------------------------------

class TestGoodsAverage:
    def setup_method(self):
        self.market = MarketMechanism(goods_clearing_mode="average")

    def _clear(self, firms, hh, avg_price=10.0):
        return self.market.clear_goods_market(firms, hh, avg_price)

    def test_demand_limited(self):
        """Household can't afford all inventory → sells only what budget allows."""
        firms = [make_firm("f1", 1000, 5, 10, 20, inventory=20)]
        hh    = [make_household("h1", 100, 0, budget=50)]  # can buy 5 units @ price=10

        sold = self._clear(firms, hh)

        assert sold == pytest.approx(5.0)
        assert hh[0].last_bought == pytest.approx(5.0)
        assert hh[0].last_spent  == pytest.approx(50.0)
        assert hh[0].money       == pytest.approx(50.0)
        assert firms[0].inventory == pytest.approx(15.0)

    def test_supply_limited(self):
        """Household wants more than available → all inventory sold."""
        firms = [make_firm("f1", 1000, 5, 10, 20, inventory=3)]
        hh    = [make_household("h1", 500, 0, budget=500)]

        sold = self._clear(firms, hh)

        assert sold == pytest.approx(3.0)
        assert firms[0].inventory == pytest.approx(0.0)

    def test_firm_revenue(self):
        """Firm receives avg_price × units_sold."""
        firms = [make_firm("f1", 0, 0, 10, 20, inventory=10)]
        hh    = [make_household("h1", 1000, 0, budget=100)]

        self._clear(firms, hh)

        assert firms[0].money == pytest.approx(100.0)

    def test_production_target_cap(self):
        """Production is capped by production_target even if labor allows more."""
        firms = [make_firm("f1", 1000, 5, 10, production_target=3, labor_hired=10)]
        hh    = [make_household("h1", 500, 0, budget=500)]

        sold = self._clear(firms, hh)

        # Produces min(10*1.0, 3) = 3, then sells all 3
        assert sold == pytest.approx(3.0)

    def test_productivity_multiplier(self):
        """Higher productivity → more output per worker."""
        firms = [make_firm("f1", 1000, 5, 10, production_target=100,
                           labor_hired=5, productivity=2.0)]
        hh    = [make_household("h1", 10000, 0, budget=10000)]

        sold = self._clear(firms, hh)

        # Produces min(5*2.0, 100) = 10
        assert sold == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Goods market — queue mode
# ---------------------------------------------------------------------------

class TestGoodsQueue:
    def setup_method(self):
        self.market = MarketMechanism(goods_clearing_mode="queue")

    def _clear(self, firms, hh):
        return self.market.clear_goods_market(firms, hh, avg_price=0.0)

    def test_buyer_goes_cheap_first(self):
        """Household exhausts cheap firm before visiting expensive one."""
        cheap     = make_firm("cheap", 0, 0, price=5,  production_target=100, inventory=10)
        expensive = make_firm("exp",   0, 0, price=20, production_target=100, inventory=10)
        hh        = [make_household("h1", 1000, 0, budget=50)]  # buys 10 @ price=5

        self._clear([cheap, expensive], hh)

        assert cheap.inventory     == pytest.approx(0.0)
        assert expensive.inventory == pytest.approx(10.0)

    def test_spills_over_to_expensive(self):
        """After cheap firm runs out, buyer continues to expensive firm."""
        cheap     = make_firm("cheap", 0, 0, price=5,  production_target=100, inventory=4)
        expensive = make_firm("exp",   0, 0, price=10, production_target=100, inventory=10)
        hh        = [make_household("h1", 1000, 0, budget=60)]
        # 4 units @ 5 = 20 spent, 40 left → 4 more @ 10 = 40

        self._clear([cheap, expensive], hh)

        assert cheap.inventory     == pytest.approx(0.0)
        assert expensive.inventory == pytest.approx(6.0)

    def test_total_sold_returned(self):
        firms = [make_firm("f1", 0, 0, price=10, production_target=100, inventory=5)]
        hh    = [make_household("h1", 1000, 0, budget=30)]  # buys 3

        total = self._clear(firms, hh)

        assert total == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Goods market — weighted mode
# ---------------------------------------------------------------------------

class TestGoodsWeighted:
    def setup_method(self):
        self.market = MarketMechanism(goods_clearing_mode="weighted")

    def _clear(self, firms, hh):
        return self.market.clear_goods_market(firms, hh, avg_price=0.0)

    def test_weight_proportional_to_inverse_price(self):
        """Cheaper firm gets a larger share of demand."""
        cheap     = make_firm("cheap", 0, 0, price=5,  production_target=100, inventory=100)
        expensive = make_firm("exp",   0, 0, price=10, production_target=100, inventory=100)
        hh        = [make_household("h1", 1000, 0, budget=150)]

        self._clear([cheap, expensive], hh)

        # weights: cheap=1/5, exp=1/10 → normalized 2/3, 1/3
        # cheap: 150*2/3 / 5 = 20 units; exp: 150*1/3 / 10 = 5 units
        assert cheap.goods_sold     == pytest.approx(20.0)
        assert expensive.goods_sold == pytest.approx(5.0)

    def test_supply_constrained_no_spillover(self):
        """If one firm runs out, excess demand is lost (no spillover within a step)."""
        scarce = make_firm("scarce", 0, 0, price=5,  production_target=100, inventory=2)
        other  = make_firm("other",  0, 0, price=10, production_target=100, inventory=100)
        hh     = [make_household("h1", 1000, 0, budget=150)]

        self._clear([scarce, other], hh)

        assert scarce.goods_sold == pytest.approx(2.0)

    def test_household_spent_reflects_fill_rate(self):
        """Household money decreases by exactly what was spent."""
        f1 = make_firm("f1", 0, 0, price=10, production_target=100, inventory=100)
        hh = [make_household("h1", 500, 0, budget=100)]

        self._clear([f1], hh)

        assert hh[0].money      == pytest.approx(500.0 - hh[0].last_spent)
        assert hh[0].last_spent == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Invalid clearing mode
# ---------------------------------------------------------------------------

def test_invalid_clearing_mode():
    with pytest.raises(ValueError, match="Unknown goods_clearing_mode"):
        MarketMechanism(goods_clearing_mode="bogus")
