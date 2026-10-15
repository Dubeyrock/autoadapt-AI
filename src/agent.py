"""Agent controller: picks the next adaptation experiment from history, enforces the
guardrail, checks stop rules after every trial, and logs every decision.
Uses VALIDATION sets only. The test set is never touched here."""
import json
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from actions import ACTIONS, SEEDS, Context, run_action
from evaluate import mean_ci
from stop_rules import (StopConfig, budget_exhausted, diminishing_returns, gain_per_label,
                        guardrail_breached, guardrail_exhausted, paired_diff_ci, plateau,
                        significantly_better)

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

# Cheap-first screening order (by head complexity). Also used as a deterministic tie-break.
SCREEN_ORDER = ["zero_shot", "logreg_C1", "logreg_C10", "logreg_C100", "svm_C1",
                "warmstart_gentle", "warmstart_strong", "hybrid_C10", "mlp"]
assert set(SCREEN_ORDER) == set(ACTIONS), "SCREEN_ORDER must list every action"


def run_trial(ctx, name, n):
    """One trial = one (action, n_labels) config over all seeds. Shared with compare.py later."""
    seeds = [0] if name == "zero_shot" else list(SEEDS)
    rows = [run_action(ctx, name, n, s) for s in seeds]
    f1s = [float(r["macro_f1"]) for r in rows]
    mean, lo, hi = mean_ci(f1s)
    avg = lambda key: float(np.mean([r[key] for r in rows]))
    return {
        "action": name, "n_labels": int(rows[0]["n_labels"]), "seeds": seeds,
        "f1_per_seed": f1s, "f1_mean": mean, "f1_ci": [lo, hi],
        "precision": avg("precision"), "recall": avg("recall"),
        "src_drop": avg("src_drop"), "train_sec": avg("train_sec"), "infer_ms": avg("infer_ms"),
        "guardrail_applies": bool(rows[0]["guardrail_applies"]),
    }


class Agent:
    def __init__(self, ctx, cfg=None, log_path=None, ladder=(50, 100, 200)):
        self.ctx = ctx
        self.cfg = cfg or StopConfig()
        self.ladder = tuple(ladder)
        self.log_path = Path(log_path) if log_path else LOG_DIR / "trials.jsonl"
        self.history = []
        self.banned = {}          # action -> reason
        self.frozen = set()       # actions no longer escalated (diminishing returns)
        self.finalists = None
        self.challengers = []
        self.t0 = None

    # ---------- logging ----------
    def _log(self, entry):
        entry = {"t_sec": round(time.perf_counter() - self.t0, 2), **entry}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=float) + "\n")

    def _decision(self, kind, **info):
        self._log({"type": "decision", "kind": kind, **info})
        print(f"    -> {kind}: " + ", ".join(f"{k}={v}" for k, v in info.items()))

    # ---------- state helpers ----------
    def _trial(self, name, n):
        for t in self.history:
            if t["action"] == name and t["n_labels"] == n:
                return t
        return None

    def incumbent(self):
        ok = [t for t in self.history if not t["banned"]]
        return max(ok, key=lambda t: t["f1_mean"]) if ok else None

    def _previous_rung(self, name, n):
        earlier = [t for t in self.history if t["action"] == name and t["n_labels"] < n]
        return max(earlier, key=lambda t: t["n_labels"]) if earlier else None

    # ---------- policy ----------
    def _pick_finalists(self):
        base = self._trial("zero_shot", 0)
        useful, dropped = [], []
        for t in self.history:
            if t["phase"] != "screen" or t["action"] == "zero_shot" or t["banned"]:
                continue
            (useful if significantly_better(t["f1_per_seed"], base["f1_per_seed"]) else dropped).append(t)
        useful.sort(key=lambda t: -t["f1_mean"])
        self.finalists = [t["action"] for t in useful[:2]]
        self.challengers = [t["action"] for t in useful[2:]]
        self._decision(
            "screen_summary",
            finalists=self.finalists, challengers=self.challengers,
            dropped_not_better_than_zero_shot=[t["action"] for t in dropped],
            banned=sorted(self.banned),
        )

    def _next(self):
        # Phase A: screen every action once at the cheapest rung (zero-shot needs no labels)
        for name in SCREEN_ORDER:
            n = 0 if name == "zero_shot" else self.ladder[0]
            if self._trial(name, n) is None:
                return name, n, "screen", "cheap-first screening"
        if self.finalists is None:
            self._pick_finalists()
        # Phase B: escalate the top-2 finalists rung by rung (round-robin)
        for i, rung in enumerate(self.ladder[1:], start=1):
            for name in self.finalists:
                if name in self.frozen:
                    continue
                if self._trial(name, rung) is None and self._trial(name, self.ladder[i - 1]) is not None:
                    return name, rung, "escalate", f"finalist, rung {self.ladder[i - 1]}->{rung}"
        # Phase C: check the pruned actions at the top rung (does the 50-label pruning hold?)
        top = self.ladder[-1]
        for name in self.challengers:
            if self._trial(name, top) is None:
                return name, top, "challenge", "pruning check at top rung"
        return None

    # ---------- one trial ----------
    def _execute(self, name, n, phase, why):
        trial = run_trial(self.ctx, name, n)
        banned = trial["guardrail_applies"] and guardrail_breached(trial["src_drop"], self.cfg)
        inc = self.incumbent()
        if banned:
            improved, diff = False, None
        elif inc is None:
            improved, diff = True, None
        else:
            diff = paired_diff_ci(trial["f1_per_seed"], inc["f1_per_seed"])
            improved = diff[1] > 0
        trial.update({
            "type": "trial", "trial_id": len(self.history) + 1, "phase": phase, "why": why,
            "banned": bool(banned), "improved": bool(improved),
            "incumbent_before": f"{inc['action']}@{inc['n_labels']}" if inc else None,
            "diff_vs_incumbent": list(diff) if diff else None,
        })
        self.history.append(trial)
        self._log(trial)
        lo, hi = trial["f1_ci"]
        print(f"#{trial['trial_id']:02d} [{phase:8s}] {name:17s} n={trial['n_labels']:3d}  "
              f"F1={trial['f1_mean']:.3f} [{lo:.3f}, {hi:.3f}]  src_drop={trial['src_drop']:.3f}  "
              f"improved={improved}  banned={banned}")

        if banned:
            self.banned[name] = f"src_drop {trial['src_drop']:.3f} > {self.cfg.guardrail_max_drop}"
            self._decision("ban", action=name, src_drop=round(trial["src_drop"], 3),
                           limit=self.cfg.guardrail_max_drop)
        if phase == "escalate":
            prev = self._previous_rung(name, trial["n_labels"])
            g = gain_per_label(prev, trial)
            if diminishing_returns(prev, trial, self.cfg):
                self.frozen.add(name)
                self._decision("freeze", action=name, gain_per_label=round(g, 5),
                               threshold=self.cfg.min_gain_per_label)
        return trial

    def _check_stop(self):
        r = budget_exhausted(len(self.history), time.perf_counter() - self.t0, self.cfg)
        if r:
            return r
        if guardrail_exhausted([a for a in ACTIONS if a != "zero_shot"], self.banned):
            return "guardrail_all_banned"
        return plateau(self.history, self.cfg)

    # ---------- final selection ----------
    def select_final(self):
        """Parsimony rule: among trials not significantly worse than the top-mean trial,
        choose the one using the fewest labels (cheap-first order breaks ties deterministically)."""
        ok = [t for t in self.history if not t["banned"]]
        top = max(ok, key=lambda t: t["f1_mean"])
        tied = [t for t in ok if paired_diff_ci(t["f1_per_seed"], top["f1_per_seed"])[2] >= 0]
        chosen = min(tied, key=lambda t: (t["n_labels"], SCREEN_ORDER.index(t["action"])))
        return top, chosen, tied

    # ---------- main loop ----------
    def run(self):
        self.log_path.parent.mkdir(exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")
        self.t0 = time.perf_counter()
        self._log({"type": "config", "cfg": asdict(self.cfg), "ladder": list(self.ladder),
                   "seeds": SEEDS, "src_f1": self.ctx.src_f1})
        reason = None
        while reason is None:
            nxt = self._next()
            if nxt is None:
                reason = "search_space_exhausted"
                break
            self._execute(*nxt)
            reason = self._check_stop()

        top, chosen, tied = self.select_final()
        brief = lambda t: {"action": t["action"], "n_labels": t["n_labels"],
                           "f1_mean": t["f1_mean"], "f1_ci": t["f1_ci"], "train_sec": t["train_sec"]}
        elapsed = time.perf_counter() - self.t0
        summary = {
            "stop_reason": reason, "n_trials": len(self.history), "elapsed_sec": round(elapsed, 2),
            "selected": brief(chosen), "top_mean": brief(top),
            "indistinguishable_from_top": [f"{t['action']}@{t['n_labels']}" for t in tied],
            "banned": self.banned, "frozen": sorted(self.frozen),
            "finalists": self.finalists, "challengers": self.challengers,
            "config": asdict(self.cfg),
            "note": "Selection used validation data only. Test set not touched.",
        }
        self._log({"type": "stop", "reason": reason, "n_trials": len(self.history)})
        (self.log_path.parent / "agent_summary.json").write_text(
            json.dumps(summary, indent=2, default=float), encoding="utf-8")

        print(f"\nSTOP: {reason}  after {len(self.history)} trials, {elapsed:.1f}s")
        print(f"Top mean     : {top['action']}@{top['n_labels']}  F1={top['f1_mean']:.3f}")
        print(f"Selected     : {chosen['action']}@{chosen['n_labels']}  F1={chosen['f1_mean']:.3f}  "
              f"(fewest labels among {len(tied)} trials not significantly worse than top)")
        print(f"Banned       : {sorted(self.banned)}")
        print(f"Saved        : {self.log_path}  and  agent_summary.json")
        return summary


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    Agent(Context()).run()