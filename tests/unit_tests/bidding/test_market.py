import numpy as np
from easy_marl.examples.bidding.market import market_clearing, sigmoid


class TestMarketClearing:
    """Tests for the market_clearing function."""

    def test_basic_clearing(self):
        """Test a simple case where demand is met by the first cheap generator."""
        # 3 Agents
        # Agent 0: bid 10, qty 100
        # Agent 1: bid 20, qty 100
        # Agent 2: bid 30, qty 100
        # Demand: 150

        bids = np.array([10.0, 20.0, 30.0])
        quantities = np.array([100.0, 100.0, 100.0])
        demand = 150.0

        price, dispatched = market_clearing(bids, quantities, demand)

        # Expected:
        # Agent 0 (10$) fully cleared -> 100
        # Agent 1 (20$) partially cleared -> 50 (150 - 100)
        # Agent 2 (30$) not cleared -> 0
        # Clearing price -> 20.0

        assert price == 20.0
        np.testing.assert_allclose(dispatched, [100.0, 50.0, 0.0])

    def test_all_cleared(self):
        """Test case where demand equals total supply."""
        bids = np.array([10.0, 20.0])
        quantities = np.array([50.0, 50.0])
        demand = 100.0

        price, dispatched = market_clearing(bids, quantities, demand)

        # Marginal generator is the last one (20$)
        assert price == 20.0
        np.testing.assert_allclose(dispatched, [50.0, 50.0])

    def test_shortage(self):
        """Test case where demand exceeds total supply."""
        bids = np.array([10.0, 20.0])
        quantities = np.array([50.0, 50.0])
        demand = 150.0

        price, dispatched = market_clearing(bids, quantities, demand)

        # Price should be cap (max bid in this implementation, or typically Value of Lost Load)
        # The implementation returns the highest bid as price if demand > supply
        assert price == 20.0
        np.testing.assert_allclose(dispatched, [50.0, 50.0])

    def test_zero_demand(self):
        """Test with zero demand."""
        bids = np.array([10.0, 20.0])
        quantities = np.array([50.0, 50.0])
        demand = 0.0

        price, dispatched = market_clearing(bids, quantities, demand)

        # Price logic with 0 demand:
        # If m=0, code does: P_t = bids_sorted[0]
        # This implies price is set by the cheapest generator even if no one runs.
        # This is a specific design choice in the current code (or a bug? but we test consistent with code).
        assert price == 10.0
        np.testing.assert_allclose(dispatched, [0.0, 0.0])

    def test_unsorted_inputs(self):
        """Test that function handles unsorted bids correctly."""
        # Unsorted: [30, 10, 20]
        bids = np.array([30.0, 10.0, 20.0])
        quantities = np.array([100.0, 100.0, 100.0])
        demand = 150.0

        price, dispatched = market_clearing(bids, quantities, demand)

        # Sorted order: 10 (idx 1), 20 (idx 2), 30 (idx 0)
        # Demand 150 -> 100 from idx 1, 50 from idx 2, 0 from idx 0
        # Price 20.0

        assert price == 20.0
        # Order in output should match input indices
        # idx 0 (30$): 0
        # idx 1 (10$): 100
        # idx 2 (20$): 50
        np.testing.assert_allclose(dispatched, [0.0, 100.0, 50.0])


class TestSigmoid:
    """Tests for helper functions."""

    def test_sigmoid_values(self):
        assert sigmoid(0) == 0.5
        assert sigmoid(100) > 0.99
        assert sigmoid(-100) < 0.01
