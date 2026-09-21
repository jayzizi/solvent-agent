"""tests/test_cogs_accounting.py — the cost side of the ledger is measured, not assumed.

Two regressions, both of which biased reported margin upward:

* `research_brief` rebound `usage` on every round of the tool-calling loop and
  returned only the last one, so planning rounds and `summarize` tool calls
  were billed to nobody.
* `reconcile_cogs` only warned when actual cost *exceeded* the estimate, so a
  quote that assumed resources the job never used looked indistinguishable
  from an accurate one.
"""

import unittest
from unittest import mock

from solvent import nemotron, service
from solvent.pricing import Quote

USAGE = {"prompt_tokens": 700, "completion_tokens": 300, "total_tokens": 1_000}


def _quote(price_cents: int, est_cost_cents: int) -> Quote:
    margin = price_cents - est_cost_cents
    return Quote(
        price_cents=price_cents,
        est_cost_cents=est_cost_cents,
        margin_cents=margin,
        margin_pct=round(100 * margin / price_cents, 1),
        accept=True,
        reason="accepted",
        cost_breakdown={},
    )


class TestTokenAccounting(unittest.TestCase):
    def _run_with_rounds(self, replies):
        """Drive research_brief through a scripted sequence of model replies."""
        calls = []

        def fake_complete(system, user):
            calls.append(1)
            index = min(len(calls) - 1, len(replies) - 1)
            return replies[index], dict(USAGE)

        with mock.patch.object(nemotron, "complete", fake_complete):
            _text, usage, ctx = nemotron.research_brief("test topic", "ctx")
        return len(calls), usage, ctx

    def test_single_round_bills_one_call(self):
        n_calls, usage, _ctx = self._run_with_rounds(["# Brief\n## Findings\nDone."])
        self.assertEqual(n_calls, 1)
        self.assertEqual(usage["total_tokens"], 1_000)

    def test_tool_rounds_are_billed_not_discarded(self):
        n_calls, usage, ctx = self._run_with_rounds(
            [
                '<tool_call>{"name":"market_data","arguments":{"ticker":"NVDA"}}</tool_call>',
                '<tool_call>{"name":"web_search","arguments":{"query":"edge ai"}}</tool_call>',
                "# Brief\n## Findings\nDone.",
            ]
        )
        self.assertEqual(ctx.market_data_calls, 1)
        self.assertEqual(ctx.web_search_calls, 1)
        # Previously this returned 1_000 — only the final synthesis pass.
        self.assertEqual(usage["total_tokens"], n_calls * 1_000)
        self.assertGreater(n_calls, 1)

    def test_every_usage_component_accumulates(self):
        n_calls, usage, _ctx = self._run_with_rounds(
            [
                '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
                "# Brief\n## Findings\nDone.",
            ]
        )
        self.assertEqual(usage["prompt_tokens"], n_calls * 700)
        self.assertEqual(usage["completion_tokens"], n_calls * 300)
        self.assertEqual(usage["total_tokens"], n_calls * 1_000)

    def test_billed_tokens_drive_the_vendor_charge(self):
        _n, usage, ctx = self._run_with_rounds(
            [
                '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
                "# Brief\n## Findings\nDone.",
            ]
        )
        resources = service._resources_from_usage(usage, ctx)
        nemotron_charge = next(a for v, a, _ in resources if v == "nvidia-nemotron")
        self.assertGreater(nemotron_charge, 0)
        # 30c per 1k tokens against the accumulated total, not the last call.
        self.assertEqual(nemotron_charge, round(usage["total_tokens"] / 1_000 * 30))


class TestCogsDriftIsSymmetric(unittest.TestCase):
    def test_large_over_estimate_is_flagged(self):
        # The shipped demo case: quoted $6.19, actually cost $0.55.
        cogs = service.reconcile_cogs(_quote(4_900, 619), {"actual_cost_cents": 55})
        self.assertTrue(cogs["cost_warning"])
        self.assertEqual(cogs["cost_drift_direction"], "under")
        self.assertEqual(cogs["margin_drift_cents"], -564)

    def test_large_under_estimate_is_still_flagged(self):
        cogs = service.reconcile_cogs(_quote(4_900, 619), {"actual_cost_cents": 1_200})
        self.assertTrue(cogs["cost_warning"])
        self.assertEqual(cogs["cost_drift_direction"], "over")

    def test_accurate_estimate_is_not_flagged(self):
        cogs = service.reconcile_cogs(_quote(4_900, 619), {"actual_cost_cents": 630})
        self.assertFalse(cogs["cost_warning"])
        self.assertEqual(cogs["cost_drift_direction"], "over")

    def test_exact_estimate_reports_no_drift(self):
        cogs = service.reconcile_cogs(_quote(4_900, 619), {"actual_cost_cents": 619})
        self.assertFalse(cogs["cost_warning"])
        self.assertEqual(cogs["cost_drift_direction"], "none")
        self.assertEqual(cogs["margin_drift_cents"], 0)

    def test_actual_margin_is_computed_from_actual_cost(self):
        cogs = service.reconcile_cogs(_quote(4_900, 619), {"actual_cost_cents": 55})
        self.assertEqual(cogs["actual_margin_pct"], round(100 * (4_900 - 55) / 4_900, 1))


if __name__ == "__main__":
    unittest.main()
