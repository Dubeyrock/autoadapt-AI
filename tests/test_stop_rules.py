from stop_rules import (StopConfig, budget_exhausted, diminishing_returns, guardrail_breached,
                        guardrail_exhausted, paired_diff_ci, plateau, significantly_better)

CFG = StopConfig()


def T(phase, improved):
    return {"phase": phase, "improved": improved}


def test_plateau_ignores_screening_trials():
    assert plateau([T("screen", False)] * 9, CFG) is None


def test_plateau_needs_patience_trials():
    assert plateau([T("screen", False)] * 9 + [T("escalate", False)] * 2, CFG) is None


def test_plateau_fires_after_three_non_improving():
    hist = [T("screen", True)] * 9 + [T("escalate", False)] * 3
    assert plateau(hist, CFG) == "plateau"


def test_plateau_not_fired_if_one_recent_trial_improved():
    hist = [T("screen", True)] * 9 + [T("escalate", False), T("escalate", True), T("challenge", False)]
    assert plateau(hist, CFG) is None


def test_budget_rules():
    assert budget_exhausted(20, 1.0, CFG) == "budget_trials"
    assert budget_exhausted(3, 601.0, CFG) == "budget_time"
    assert budget_exhausted(3, 1.0, CFG) is None


def test_diminishing_returns():
    prev = {"f1_mean": 0.90, "n_labels": 50}
    assert diminishing_returns(prev, {"f1_mean": 0.905, "n_labels": 100}, CFG)       # 0.0001/label
    assert not diminishing_returns(prev, {"f1_mean": 0.93, "n_labels": 100}, CFG)    # 0.0006/label


def test_guardrail():
    assert guardrail_breached(0.031, CFG) and not guardrail_breached(0.03, CFG)
    assert guardrail_exhausted(["a", "b"], ["a", "b"])
    assert not guardrail_exhausted(["a", "b"], ["a"])


def test_paired_significance():
    a = [0.90, 0.91, 0.92, 0.93, 0.94]
    b = [0.85, 0.86, 0.86, 0.88, 0.89]
    assert significantly_better(a, b)
    assert not significantly_better(a, a)
    assert significantly_better(a, [0.55])        # zero-shot (1 value) is broadcast
    assert paired_diff_ci(a, b)[0] > 0