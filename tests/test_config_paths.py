"""tests/test_config_paths.py — configuration resolves from the app home, not cwd.

Regression cover for the fail-open defect: the spend policy, pricing
overrides, Telegram allowlist and rate-limit store were all bound to a
cwd-relative `./.solvent`, while the treasury resolved through
`$SOLVENT_HOME`. Starting the agent from a different directory therefore kept
the ledger continuous while silently restoring the permissive built-in spend
limits.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from solvent import paths
from solvent.guardrails import load_spend_policy, policy_override_path

TIGHT_POLICY = {
    "daily_budget_cents": 500,
    "per_vendor_daily_cents": 100,
    "max_txn_cents": 50,
}


class TestConfigDir(unittest.TestCase):
    def test_config_dir_sits_under_solvent_home(self):
        with tempfile.TemporaryDirectory() as home:
            with mock.patch.dict(os.environ, {"SOLVENT_HOME": home}):
                self.assertEqual(paths.config_dir(), Path(home).resolve() / ".solvent")

    def test_config_dir_does_not_nest_inside_a_dot_solvent_home(self):
        # The `pip install` default home is already `~/.solvent`; nesting a
        # second `.solvent` inside it would be surprising.
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / ".solvent"
            home.mkdir()
            with mock.patch.dict(os.environ, {"SOLVENT_HOME": str(home)}):
                self.assertEqual(paths.config_dir(), home.resolve())


class TestSpendPolicyIsIndependentOfCwd(unittest.TestCase):
    """The core of the fail-open bug."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.addCleanup(lambda: os.chdir(self._cwd))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.home = Path(self._tmp.name) / "home"
        (self.home / ".solvent").mkdir(parents=True)
        (self.home / ".solvent" / "spend_policy.json").write_text(
            json.dumps(TIGHT_POLICY), encoding="utf-8"
        )
        self.elsewhere = Path(self._tmp.name) / "elsewhere"
        self.elsewhere.mkdir()
        self._env = mock.patch.dict(os.environ, {"SOLVENT_HOME": str(self.home)})
        self._env.start()
        self.addCleanup(self._env.stop)

    def test_policy_applies_from_an_unrelated_working_directory(self):
        os.chdir(self.elsewhere)
        policy = load_spend_policy()
        self.assertEqual(policy.daily_budget_cents, 500)
        self.assertEqual(policy.max_txn_cents, 50)
        self.assertEqual(policy.per_vendor_daily_cents, 100)

    def test_policy_is_the_same_from_every_directory(self):
        os.chdir(self.elsewhere)
        from_elsewhere = load_spend_policy()
        os.chdir(self.home)
        from_home = load_spend_policy()
        self.assertEqual(from_elsewhere.daily_budget_cents, from_home.daily_budget_cents)
        self.assertEqual(from_elsewhere.max_txn_cents, from_home.max_txn_cents)

    def test_absent_policy_still_falls_back_to_defaults(self):
        (self.home / ".solvent" / "spend_policy.json").unlink()
        os.chdir(self.elsewhere)
        policy = load_spend_policy()
        self.assertEqual(policy.daily_budget_cents, 25_000)

    def test_policy_override_path_points_into_the_app_home(self):
        os.chdir(self.elsewhere)
        self.assertEqual(
            policy_override_path(), (self.home / ".solvent" / "spend_policy.json").resolve()
        )


class TestLegacyFallback(unittest.TestCase):
    """A pre-existing `./.solvent` file is still honoured, but cannot widen."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.addCleanup(lambda: os.chdir(self._cwd))
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_legacy_cwd_file_is_found_when_the_app_home_has_none(self):
        workdir = Path(self._tmp.name) / "legacy"
        (workdir / ".solvent").mkdir(parents=True)
        (workdir / ".solvent" / "spend_policy.json").write_text(
            json.dumps(TIGHT_POLICY), encoding="utf-8"
        )
        home = Path(self._tmp.name) / "home"
        home.mkdir()
        with mock.patch.dict(os.environ, {"SOLVENT_HOME": str(home)}):
            os.chdir(workdir)
            self.assertEqual(load_spend_policy().daily_budget_cents, 500)

    def test_app_home_wins_over_a_legacy_cwd_file(self):
        workdir = Path(self._tmp.name) / "legacy2"
        (workdir / ".solvent").mkdir(parents=True)
        (workdir / ".solvent" / "spend_policy.json").write_text(
            json.dumps({"daily_budget_cents": 99_999}), encoding="utf-8"
        )
        home = Path(self._tmp.name) / "home2"
        (home / ".solvent").mkdir(parents=True)
        (home / ".solvent" / "spend_policy.json").write_text(
            json.dumps(TIGHT_POLICY), encoding="utf-8"
        )
        with mock.patch.dict(os.environ, {"SOLVENT_HOME": str(home)}):
            os.chdir(workdir)
            # The canonical location is authoritative; a stray file in the
            # working directory must not loosen the limits in force.
            self.assertEqual(load_spend_policy().daily_budget_cents, 500)


if __name__ == "__main__":
    unittest.main()
