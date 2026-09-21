import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solvent.pricing import (
    RESOURCE_COSTS_CENTS,
    PricingPolicy,
    estimate_cost,
    get_resource_costs,
    quote,
)

from .pricing_fixture import TEST_PRICING, cost_of


class TestPricing(unittest.TestCase):
    """Unit tests for the pricing logic and margin gate rules of SOLVENT.

    Costs are pinned to `tests/pricing_fixture` (30c per 1k tokens) so these
    assertions test the gate, not a vendor's current price list. Only
    inference costs anything under the measured table — the market-data and
    web-search endpoints are free and the PDF/email vendors do not exist — so
    every total below is a function of `est_tokens` alone.
    """

    def setUp(self) -> None:
        patcher = mock.patch(
            "solvent.providers.active_pricing", return_value=TEST_PRICING
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_estimate_cost_defaults(self) -> None:
        """Default job: 8000 tokens, and nothing else that costs money."""
        total_cost, breakdown = estimate_cost({})

        self.assertEqual(breakdown["llm_inference"], cost_of(8_000))  # 240c
        self.assertEqual(breakdown["market_data"], 0)  # stooq.com is free
        self.assertEqual(breakdown["web_search"], 0)  # DuckDuckGo is free
        self.assertEqual(breakdown["pdf_render"], 0)  # no such service
        self.assertEqual(breakdown["email_send"], 0)  # SMTP you already run
        self.assertEqual(total_cost, cost_of(8_000))

    def test_estimate_cost_custom(self) -> None:
        """Scope dials that cost nothing do not move the total."""
        job = {"est_tokens": 12_500, "market_data_calls": 5, "web_search_calls": 10}
        total_cost, breakdown = estimate_cost(job)

        self.assertEqual(breakdown["llm_inference"], cost_of(12_500))  # 375c
        self.assertEqual(total_cost, cost_of(12_500))

    def test_free_resources_do_not_change_the_total(self) -> None:
        """Ten market-data pulls cost exactly as much as none: nothing."""
        base = {"est_tokens": 10_000, "market_data_calls": 0, "web_search_calls": 0}
        rich = {"est_tokens": 10_000, "market_data_calls": 10, "web_search_calls": 30}
        self.assertEqual(estimate_cost(base)[0], estimate_cost(rich)[0])

    def test_only_tokens_drive_cost(self) -> None:
        """Doubling the tokens doubles the bill; nothing else does."""
        single = estimate_cost({"est_tokens": 10_000})[0]
        double = estimate_cost({"est_tokens": 20_000})[0]
        self.assertEqual(double, 2 * single)

    def test_quote_below_minimum_price(self) -> None:
        """Test that a quote is declined if the budget is below the policy minimum price."""
        policy = PricingPolicy(min_price_cents=1500)
        job = {
            "budget_cents": 1000,  # $10 budget, less than $15 minimum
            "est_tokens": 2000,
            "market_data_calls": 1,
            "web_search_calls": 2,
        }
        q = quote(job, policy)
        self.assertFalse(q.accept)
        self.assertIn("below minimum order size", q.reason)

    def test_quote_below_fulfilment_cost(self) -> None:
        """Test that a quote is declined if the budget is below the estimated cost."""
        policy = PricingPolicy(min_price_cents=100)
        # 8000 tokens = 240c of inference; nothing else costs anything.
        job = {
            "budget_cents": 200,  # clears the minimum order size, under the 240c cost
            "est_tokens": 8000,
            "market_data_calls": 5,
            "web_search_calls": 10,
        }
        q = quote(job, policy)
        self.assertFalse(q.accept)
        self.assertEqual(q.reason, "customer budget below fulfilment cost")

    def test_quote_below_margin_floor(self) -> None:
        """Test that a quote is declined if the margin is below the floor percentage."""
        policy = PricingPolicy(min_price_cents=100, margin_floor_pct=35.0)
        # Cost 240c, budget 300c -> margin 60c = 20.0%, under the 35% floor.
        job = {
            "budget_cents": 300,
            "est_tokens": 8000,
            "market_data_calls": 5,
            "web_search_calls": 10,
        }
        q = quote(job, policy)
        self.assertFalse(q.accept)
        self.assertIn("below floor", q.reason)

    def test_quote_accepted(self) -> None:
        """Test a successful quote that meets all pricing gate criteria."""
        policy = PricingPolicy(min_price_cents=1500, margin_floor_pct=35.0)
        # Cost 240c, budget 2000c -> margin 1760c = 88.0%.
        job = {
            "budget_cents": 2000,
            "est_tokens": 8000,
            "market_data_calls": 5,
            "web_search_calls": 10,
        }
        q = quote(job, policy)
        self.assertTrue(q.accept)
        self.assertEqual(q.reason, "accepted")
        self.assertEqual(q.price_cents, 2000)
        self.assertEqual(q.est_cost_cents, cost_of(8_000))
        self.assertEqual(q.margin_cents, 2000 - cost_of(8_000))
        self.assertEqual(q.margin_pct, 88.0)

    def test_quote_zero_budget(self) -> None:
        """Test a job with zero budget."""
        job = {"budget_cents": 0}
        q = quote(job)
        self.assertFalse(q.accept)
        self.assertEqual(q.margin_pct, -100.0)


class TestResourceCostsOverrides(unittest.TestCase):
    """Coverage for get_resource_costs() loading .solvent/pricing_overrides.json."""

    def setUp(self) -> None:
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self._tmp.name, ".solvent"), exist_ok=True)
        os.chdir(self._tmp.name)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_defaults_when_no_override_file(self) -> None:
        costs = get_resource_costs()
        self.assertEqual(costs, RESOURCE_COSTS_CENTS)
        # Must be a copy, not the same dict object
        costs["pdf_render"] = 999
        self.assertNotEqual(RESOURCE_COSTS_CENTS["pdf_render"], 999)

    def test_measured_defaults_are_zero_for_the_free_resources(self) -> None:
        """The zeros are measurements, not unset placeholders."""
        costs = get_resource_costs()
        for key in ("market_data_call", "web_search_call", "pdf_render", "email_send"):
            self.assertEqual(costs[key], 0, key)
        # Inference is priced per-token from providers, not from this table.
        self.assertNotIn("nemotron_tokens_per_1k", costs)

    def test_valid_override_applied(self) -> None:
        """An operator whose provider really does charge can say so."""
        Path(".solvent/pricing_overrides.json").write_text(
            json.dumps({"email_send": 55, "pdf_render": 25})
        )
        costs = get_resource_costs()
        self.assertEqual(costs["email_send"], 55)
        self.assertEqual(costs["pdf_render"], 25)
        self.assertEqual(
            costs["market_data_call"],
            RESOURCE_COSTS_CENTS["market_data_call"],
        )

    def test_invalid_json_falls_back_to_defaults(self) -> None:
        Path(".solvent/pricing_overrides.json").write_text("{not valid json")
        costs = get_resource_costs()
        self.assertEqual(costs, RESOURCE_COSTS_CENTS)

    def test_non_dict_json_falls_back_to_defaults(self) -> None:
        Path(".solvent/pricing_overrides.json").write_text("[1, 2, 3]")
        costs = get_resource_costs()
        self.assertEqual(costs, RESOURCE_COSTS_CENTS)

    def test_unknown_keys_and_non_numeric_values_ignored(self) -> None:
        Path(".solvent/pricing_overrides.json").write_text(
            json.dumps({"unknown_key": 999, "web_search_call": "expensive"})
        )
        costs = get_resource_costs()
        self.assertEqual(costs["web_search_call"], RESOURCE_COSTS_CENTS["web_search_call"])
        self.assertNotIn("unknown_key", costs)


if __name__ == "__main__":
    unittest.main()
