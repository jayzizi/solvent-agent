"""tests/test_server_job_routes.py — job submission through the hosted API.

Regression cover for the event-callback cycle that made `create_app()` wire
`agent.on_event` to a function that called `agent._capture_event`, which calls
`agent.on_event`. Every emitted event recursed until the stack was exhausted,
so both job-submission routes raised `RecursionError` while the rest of the
suite stayed green — no existing test ever pushed a job through the app.
"""

import os
import unittest
from unittest import mock

from solvent.agent import Solvent
from solvent.server import create_app

TOKEN = "d" * 32


def _client():
    try:
        from fastapi.testclient import TestClient
    except ImportError:  # pragma: no cover - optional extra
        raise unittest.SkipTest("FastAPI test client is not installed")
    return TestClient(create_app(fresh=True))


class TestAgentEventFanout(unittest.TestCase):
    """The structural defect, independent of FastAPI."""

    def test_capture_event_does_not_reenter_on_event(self):
        agent = Solvent(seed_cents=10_000, fresh=True)
        seen = []
        agent.on_event = seen.append

        agent._emit(stage="test", job_id="X")

        self.assertEqual(len(seen), 1)
        self.assertEqual(len(agent.log), 1)

    def test_runner_events_reach_the_sink_exactly_once(self):
        agent = Solvent(seed_cents=10_000, fresh=True)
        seen = []
        agent.on_event = seen.append

        # The runner's own callback is `agent._capture_event`, so a stage event
        # must land in the log once and reach the sink once — not twice, and
        # not unboundedly.
        agent._runner._emit(stage="quoted", job_id="Y")

        self.assertEqual(len(seen), 1)
        self.assertEqual(len(agent.log), 1)


class TestJobRoutes(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(
            os.environ,
            {
                "SOLVENT_DASHBOARD_TOKEN": TOKEN,
                "SOLVENT_SKIP_ONBOARD": "1",
                # An accepted job mints a hosted-brief URL during checkout, so
                # the signing secret has to be present or the route 500s for a
                # reason unrelated to what these tests are pinning.
                "SOLVENT_DELIVERY_SECRET": "x" * 32,
            },
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.body = {
            "topic": "Competitive landscape for AI inference chips",
            "budget_cents": 4_900,
            "customer_email": "buyer@example.com",
            "context": "diligence",
        }

    def test_public_job_intake_does_not_crash(self):
        client = _client()
        response = client.post("/jobs", json=self.body)
        self.assertLess(response.status_code, 500, response.text)

    def test_authenticated_job_intake_does_not_crash(self):
        client = _client()
        response = client.post(
            "/api/job", json=self.body, headers={"X-Solvent-Dashboard-Token": TOKEN}
        )
        self.assertLess(response.status_code, 500, response.text)

    def test_api_job_still_requires_the_dashboard_token(self):
        client = _client()
        self.assertEqual(client.post("/api/job", json=self.body).status_code, 403)

    def test_declined_job_returns_the_decline_rather_than_crashing(self):
        # A decline is raised through `agent._emit`, the other path into the
        # sink, so it exercises the cycle independently of the stage runner.
        client = _client()
        response = client.post(
            "/jobs", json={**self.body, "budget_cents": 600, "topic": "One-line EBITDA"}
        )
        self.assertLess(response.status_code, 500, response.text)
        self.assertEqual(response.json().get("stage"), "declined")


if __name__ == "__main__":
    unittest.main()
