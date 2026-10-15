import json
import tempfile
import unittest
from pathlib import Path

from actions import Context, run_action
from agent import Agent
from embed import cache_path
from stop_rules import StopConfig

@unittest.skipUnless(
    cache_path("yt_val").exists(),
    "embedding cache missing; run python src/embed.py",
)
class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = Context()

    def test_run_action_deterministic(self):
        for name in ("logreg_C10", "hybrid_C10"):
            a = run_action(self.ctx, name, 50, 0)["macro_f1"]
            b = run_action(self.ctx, name, 50, 0)["macro_f1"]
            self.assertEqual(a, b)

    def test_warmstart_breaks_guardrail(self):
        result = run_action(self.ctx, "warmstart_gentle", 50, 0)
        self.assertTrue(
            result["guardrail_applies"]
            and result["src_drop"] > StopConfig().guardrail_max_drop
        )

    def test_agent_stops_on_trial_budget_and_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "trials.jsonl"
            summary = Agent(
                self.ctx, cfg=StopConfig(max_trials=3), log_path=log
            ).run()
            self.assertEqual(summary["stop_reason"], "budget_trials")
            self.assertEqual(summary["n_trials"], 3)
            entries = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(entries[0]["type"], "config")
            self.assertEqual(entries[-1]["type"], "stop")
            self.assertEqual(sum(e["type"] == "trial" for e in entries), 3)