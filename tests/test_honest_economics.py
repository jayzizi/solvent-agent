"""tests/test_honest_economics.py — the cost table reflects what things cost.

Deliberately does NOT use `tests/pricing_fixture`. These assertions are about
real published rates and real measurements, and they are meant to break if
someone reintroduces an invented cost.

The original table billed 30c per 1k tokens to "nvidia-nemotron", $1.20 per
pull to "market-data-api" and $0.40 per brief to "pdf-render-saas". Measured:
the market-data endpoint is stooq.com (keyless, free), web search is
DuckDuckGo's public API (keyless, free), and neither the PDF nor the email
vendor exists at all.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from solvent import providers
from solvent.pricing import estimate_cost_exact, get_resource_costs, scope_steps


class TestFreeResourcesCostNothing(unittest.TestCase):
    def test_the_free_endpoints_bill_zero(self):
        costs = get_resource_costs()
        self.assertEqual(costs["market_data_call"], 0)
        self.assertEqual(costs["web_search_call"], 0)

    def test_the_nonexistent_vendors_bill_zero(self):
        costs = get_resource_costs()
        self.assertEqual(costs["pdf_render"], 0)
        self.assertEqual(costs["email_send"], 0)

    def test_inference_is_not_a_flat_per_1k_constant(self):
        # Input and output differ ~5x; a single blended constant hid that.
        self.assertNotIn("nemotron_tokens_per_1k", get_resource_costs())


class TestPublishedRates(unittest.TestCase):
    def test_opus_5_matches_published_pricing(self):
        p = providers.PRICING["claude-opus-5"]
        self.assertEqual(p.input_cents_per_mtok, 500)  # $5.00 / MTok
        self.assertEqual(p.output_cents_per_mtok, 2_500)  # $25.00 / MTok
        self.assertTrue(p.verified)

    def test_output_is_priced_above_input(self):
        for model_id in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
            p = providers.PRICING[model_id]
            self.assertGreater(p.output_cents_per_mtok, p.input_cents_per_mtok, model_id)

    def test_unverified_rates_are_marked_not_invented(self):
        """A rate nobody checked must say so rather than look plausible."""
        for model_id in ("grok-4-6", "nvidia/llama-3.1-nemotron-ultra-253b-v1"):
            p = providers.PRICING[model_id]
            self.assertFalse(p.verified, model_id)
            self.assertIn("UNVERIFIED", p.source)

    def test_an_unknown_model_is_unverified_rather_than_guessed(self):
        with mock.patch.dict(os.environ, {"SOLVENT_MODEL": "some-model-we-never-heard-of"}):
            p = providers.active_pricing()
        self.assertFalse(p.verified)


class TestRealBriefCosts(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"SOLVENT_MODEL": "claude-opus-5"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_a_typical_brief_costs_cents_not_dollars(self):
        # J1 from SAMPLE_JOBS: 9,000 tokens. The old table quoted $6.19.
        exact, _ = estimate_cost_exact(
            {"est_tokens": 9_000, "market_data_calls": 2, "web_search_calls": 8}
        )
        self.assertLess(exact, 100)  # under a dollar
        self.assertGreater(exact, 0)

    def test_cost_is_entirely_inference(self):
        _exact, breakdown = estimate_cost_exact({"est_tokens": 9_000})
        self.assertGreater(breakdown["llm_inference"], 0)
        self.assertEqual(
            sum(v for k, v in breakdown.items() if k != "llm_inference"), 0
        )

    def test_input_and_output_are_priced_separately(self):
        p = providers.PRICING["claude-opus-5"]
        mostly_input = p.cost_cents(10_000, 0)
        mostly_output = p.cost_cents(0, 10_000)
        self.assertEqual(mostly_output, 5 * mostly_input)


class TestScopeOrderFollowsRealCost(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"SOLVENT_MODEL": "claude-opus-5"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_tokens_outrank_the_free_dials(self):
        """Narrowing a free resource buys the customer nothing, so it sorts last."""
        order = [dimension for dimension, _step in scope_steps()]
        self.assertEqual(order[0], "est_tokens")

    def test_a_zero_cost_dial_is_never_given_up_first(self):
        costs = get_resource_costs()
        order = [d for d, _ in scope_steps()]
        free = {"market_data_calls", "web_search_calls"}
        if costs["market_data_call"] == 0 and costs["web_search_call"] == 0:
            self.assertNotIn(order[0], free)


if __name__ == "__main__":
    unittest.main()
