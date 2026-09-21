"""
providers.py — what the model actually costs, per provider, in published rates.

SOLVENT's margin gate is only as honest as its cost table. The original table
in `pricing.py` was invented for the demo: it billed 30c per 1k tokens to
"nvidia-nemotron", $1.20 per pull to "market-data-api", $0.40 per brief to
"pdf-render-saas". Measured against reality:

* 30c/1k tokens is roughly 300x the real blended rate for a frontier model.
* `market-data-api` is `stooq.com` — a keyless, free CSV endpoint.
* `web-search-api` is DuckDuckGo's free public API. No credentials exist.
* `pdf-render-saas` and `email-delivery-saas` are not services. Nothing is
  called, nothing is billed, and no account exists to bill it to.

So this module holds *published, verifiable* per-token rates, split into input
and output because they differ by ~5x and a blended guess hides that. Anything
this module cannot verify is marked UNVERIFIED rather than filled with a
plausible-looking number — an invented cost is exactly the failure being
corrected here.

Rates are in **cents per million tokens** to keep the arithmetic in integers
for as long as possible; the ledger itself is integer cents.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Fraction of a brief's tokens assumed to be *output* when only a total token
#: count is known (the margin gate quotes before the job runs, so it cannot
#: know the real split). Briefs are prose-heavy but agentic loops resend the
#: transcript, which inflates input — 0.25 is a deliberate middle estimate.
#: `service.reconcile_cogs` compares this against the real split afterwards,
#: so a bad assumption shows up as drift instead of hiding.
ASSUMED_OUTPUT_SHARE = 0.25


@dataclass(frozen=True)
class ModelPricing:
    """Published per-token price for one model.

    Attributes:
        model_id: The provider's exact model identifier.
        provider: Which backend serves it.
        input_cents_per_mtok: Published input price, cents per 1M tokens.
        output_cents_per_mtok: Published output price, cents per 1M tokens.
        verified: False when the rate is a placeholder the operator must set.
            An unverified rate makes every margin figure derived from it
            untrustworthy, and `solvent guardrails` / `finance` say so.
        source: Where the number came from, so it can be re-checked.
    """

    model_id: str
    provider: str
    input_cents_per_mtok: int
    output_cents_per_mtok: int
    verified: bool = True
    source: str = ""

    def blended_cents_per_mtok(self, output_share: float = ASSUMED_OUTPUT_SHARE) -> float:
        """One rate for callers that only know a total token count."""
        share = min(max(output_share, 0.0), 1.0)
        return (1 - share) * self.input_cents_per_mtok + share * self.output_cents_per_mtok

    def cost_cents(self, input_tokens: int, output_tokens: int) -> float:
        """Exact cost of a call, in cents. Deliberately not rounded.

        A single brief can genuinely cost a fraction of a cent. Rounding here
        would either erase that (floor) or inflate it 10x (ceil), so rounding
        is left to the ledger boundary and the exact figure is reported.
        """
        return (
            input_tokens * self.input_cents_per_mtok
            + output_tokens * self.output_cents_per_mtok
        ) / 1_000_000


#: Anthropic first-party API rates. $5/$25 per MTok for Opus 5.
_CLAUDE = "https://docs.anthropic.com/en/docs/about-claude/pricing"

PRICING: dict[str, ModelPricing] = {
    "claude-opus-5": ModelPricing("claude-opus-5", "claude", 500, 2_500, True, _CLAUDE),
    "claude-sonnet-5": ModelPricing("claude-sonnet-5", "claude", 200, 1_000, True, _CLAUDE),
    "claude-haiku-4-5": ModelPricing("claude-haiku-4-5", "claude", 100, 500, True, _CLAUDE),
    # Grok / xAI and NVIDIA-hosted Nemotron: rates are NOT filled in, because
    # nothing here has verified them. Set them via SOLVENT_MODEL_PRICING (see
    # `load_pricing_override`) before trusting any margin computed from them.
    "grok-4-6": ModelPricing("grok-4-6", "grok", 0, 0, False, "UNVERIFIED — set rates"),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": ModelPricing(
        "nvidia/llama-3.1-nemotron-ultra-253b-v1", "nemotron", 0, 0, False, "UNVERIFIED — set rates"
    ),
    # The offline stub performs no network call and costs nothing. This is the
    # one zero in this file that is a measurement rather than a gap.
    "offline-stub": ModelPricing("offline-stub", "stub", 0, 0, True, "no network call"),
}

DEFAULT_MODEL = "claude-opus-5"

#: Who actually invoices for each provider's inference. The guardrail
#: allowlist is matched on these, so a new provider needs an entry here *and*
#: in `SpendPolicy.vendor_allowlist` before any spend to it can be approved.
VENDOR_BY_PROVIDER: dict[str, str] = {
    "claude": "anthropic",
    "grok": "xai",
    "nemotron": "nvidia-nemotron",
    "stub": "offline-stub",
}


def vendor_id(pricing: ModelPricing | None = None) -> str:
    """The vendor name a charge for this model should be booked against."""
    applied = pricing or active_pricing()
    return VENDOR_BY_PROVIDER.get(applied.provider, f"{applied.provider}-llm")


def active_model_id() -> str:
    """Which model SOLVENT prices against.

    Note this is the model the work is *costed* at, which is not always the
    model that runs. With no key configured the offline stub answers, and the
    stub genuinely costs nothing — but pricing the demo at zero would make the
    margin gate vacuous and tell you nothing about whether the job would pay
    for itself in production. The stub stands in for a real model, so it is
    costed as one. Set ``SOLVENT_MODEL=offline-stub`` to price at zero and
    watch the gate accept anything above the minimum order size.
    """
    explicit = os.environ.get("SOLVENT_MODEL", "").strip()
    if explicit:
        return explicit
    if os.environ.get("XAI_API_KEY"):
        return "grok-4-6"
    if os.environ.get("NVIDIA_API_KEY"):
        return "nvidia/llama-3.1-nemotron-ultra-253b-v1"
    return DEFAULT_MODEL


def active_pricing() -> ModelPricing:
    """Published rates for the active model, with operator overrides applied."""
    model_id = active_model_id()
    overrides = load_pricing_override()
    if model_id in overrides:
        return overrides[model_id]
    known = PRICING.get(model_id)
    if known is not None:
        return known
    return ModelPricing(model_id, "unknown", 0, 0, False, "UNVERIFIED — unknown model")


def load_pricing_override() -> dict[str, ModelPricing]:
    """Operator-supplied rates from ``model_pricing.json`` in the config dir.

    Shape — cents per million tokens, mirroring the published units::

        {"grok-4-6": {"input_cents_per_mtok": 200, "output_cents_per_mtok": 1000}}

    A malformed file is ignored rather than allowed to invent a rate, matching
    how `load_spend_policy` refuses to widen a policy it cannot parse.
    """
    import json

    from .paths import config_path

    path = config_path("model_pricing.json")
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}

    out: dict[str, ModelPricing] = {}
    for model_id, spec in raw.items():
        if not isinstance(model_id, str) or not isinstance(spec, dict):
            continue
        inp, outp = spec.get("input_cents_per_mtok"), spec.get("output_cents_per_mtok")
        if not _is_rate(inp) or not _is_rate(outp):
            continue
        base = PRICING.get(model_id)
        out[model_id] = ModelPricing(
            model_id=model_id,
            provider=spec.get("provider") or (base.provider if base else "custom"),
            input_cents_per_mtok=int(inp),
            output_cents_per_mtok=int(outp),
            verified=True,  # the operator asserted these
            source=str(spec.get("source") or "operator override"),
        )
    return out


def _is_rate(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
