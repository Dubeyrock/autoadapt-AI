from pathlib import Path
import unittest

import numpy as np

from data import load_all
from evaluate import sample_idx

SRC = Path(__file__).resolve().parent.parent / "src"


def _disjoint(splits, names):
    sets = [set(splits[n]["text"]) for n in names]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not (sets[i] & sets[j]), f"leakage between {names[i]} and {names[j]}"


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = load_all()

    def test_youtube_splits_disjoint_and_sized(self):
        _disjoint(self.splits, ["yt_pool", "yt_val", "yt_test"])
        self.assertEqual(len(self.splits["yt_val"]), 200)
        self.assertEqual(len(self.splits["yt_test"]), 300)

    def test_sms_splits_disjoint(self):
        _disjoint(self.splits, ["sms_train", "sms_val", "sms_test"])

    def test_sample_idx_deterministic_and_stratified(self):
        y = self.splits["yt_pool"]["y"].to_numpy()
        a, b, c = sample_idx(y, 50, 0), sample_idx(y, 50, 0), sample_idx(y, 50, 1)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(np.sort(a), np.sort(c)))
        self.assertEqual(len(set(a)), 50)
        self.assertLess(a.max(), len(y))
        self.assertLess(abs(y[a].mean() - y.mean()), 0.05)

    def test_selection_code_never_touches_test_split(self):
        filenames = ["actions.py", "agent.py", "compare.py", "baselines.py",
                     "stop_rules.py", "plots.py", "error_analysis.py"]
        for fname in filenames:
            with self.subTest(fname=fname):
                code = (SRC / fname).read_text(encoding="utf-8")
                self.assertNotIn("yt_test", code)
                self.assertNotIn("sms_test", code)