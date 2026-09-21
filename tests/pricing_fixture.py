"""tests/pricing_fixture.py — deterministic model pricing for gate-logic tests.

The margin gate's *behaviour* — declines, counter-offers, deferral — must not
depend on what a vendor currently charges. Before costs were measured, tests
could hardcode totals because the cost table was a constant in the repo; now
that inference is priced from `providers.active_pricing()`, a published price
change would silently rewrite every expected number.

So tests that exercise gate logic pin this fixture instead. The rate is chosen
to be round rather than realistic: **30c per 1,000 tokens**, input and output
alike, so a token count maps to a cost by inspection.

Tests that care about *real* economics (that a brief genuinely costs cents)
belong in `test_honest_economics.py` and must not use this fixture.
"""

from __future__ import annotations

import contextlib
from unittest import mock

from solvent.providers import ModelPricing

#: 30_000 cents per million tokens on both sides == 30c per 1k tokens, whatever
#: the input/output split turns out to be. Deliberately not a real price.
TEST_PRICING = ModelPricing(
    model_id="test-model",
    provider="claude",
    input_cents_per_mtok=30_000,
    output_cents_per_mtok=30_000,
    verified=True,
    source="test fixture — not a real rate",
)

#: Cost in cents of N tokens under the fixture, for readable assertions.
CENTS_PER_1K = 30


def cost_of(tokens: int) -> int:
    """What the fixture charges for `tokens` tokens, in whole cents."""
    return round(tokens / 1_000 * CENTS_PER_1K)


@contextlib.contextmanager
def fixed_pricing(pricing: ModelPricing = TEST_PRICING):
    """Pin `providers.active_pricing()` for the duration of the block."""
    with mock.patch("solvent.providers.active_pricing", return_value=pricing):
        yield
