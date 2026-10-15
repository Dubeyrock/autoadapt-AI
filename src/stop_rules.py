"""Stop rules and statistical helpers.

Pure functions: no file I/O and no model code, so they can be unit-tested
with synthetic trial histories. A trial is a plain dict (see agent.py) with
keys such as action, n_labels, phase, improved, banned, f1_mean, f1_per_seed, src_drop.
"""
from dataclasses import dataclass

import numpy as np

from evaluate import mean_ci


@dataclass(frozen=True)
class StopConfig:
    max_trials: int = 20
    max_seconds: float = 600.0
    patience: int = 3                  # plateau: consecutive non-improving exploit trials
    min_gain_per_label: float = 0.0002  # diminishing returns: F1 gain per extra label
    guardrail_max_drop: float = 0.03   # max allowed source (SMS) Macro-F1 drop


def paired_diff_ci(a, b):
    """Mean and 95% CI of (a - b), paired by seed. A length-1 list (zero-shot) is broadcast."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return mean_ci(d)


def significantly_better(a, b):
    return paired_diff_ci(a, b)[1] > 0


def budget_exhausted(n_trials, elapsed_sec, cfg):
    if n_trials >= cfg.max_trials:
        return "budget_trials"
    if elapsed_sec >= cfg.max_seconds:
        return "budget_time"
    return None


def plateau(history, cfg):
    """Fire if none of the last `patience` post-screening trials improved on the incumbent."""
    exploit = [t for t in history if t["phase"] != "screen"]
    if len(exploit) < cfg.patience:
        return None
    if not any(t["improved"] for t in exploit[-cfg.patience:]):
        return "plateau"
    return None


def gain_per_label(prev, cur):
    return (cur["f1_mean"] - prev["f1_mean"]) / (cur["n_labels"] - prev["n_labels"])


def diminishing_returns(prev, cur, cfg):
    return gain_per_label(prev, cur) < cfg.min_gain_per_label


def guardrail_breached(src_drop, cfg):
    return src_drop > cfg.guardrail_max_drop


def guardrail_exhausted(candidates, banned):
    """True if every candidate action has been banned by the guardrail."""
    candidates = set(candidates)
    return len(candidates) > 0 and candidates <= set(banned)