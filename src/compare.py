"""Comparison: exhaustive grid (oracle) vs agent vs random search vs fixed recipe.
Uses VALIDATION sets only. The test set is never touched here.

Run `python src/agent.py` first (this script reads logs/trials.jsonl).
Use `python src/compare.py --force` to recompute the grid cache."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from actions import Context
from agent import SCREEN_ORDER, run_trial
from stop_rules import StopConfig, paired_diff_ci

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"

CFG = StopConfig()
LADDER = (50, 100, 200)
FIXED = ("logreg_C1", 200)      # fixed recipe from the README
N_RANDOM = 500                  # number of random orderings simulated
RANDOM_SEED = 123


def key(t):
    return (t["action"], int(t["n_labels"]))


def fmt(t):
    return f"{t['action']}@{t['n_labels']}"


def compute_sec(t):
    return t["train_sec"] * len(t["seeds"])


# ---------- exhaustive grid ----------
def grid_configs():
    return [("zero_shot", 0)] + [(a, n) for a in SCREEN_ORDER if a != "zero_shot" for n in LADDER]


def build_grid(ctx, force=False):
    path = RESULTS_DIR / "grid.json"
    configs = grid_configs()
    if path.exists() and not force:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if [key(t) for t in cached] == configs:
            print(f"[cache] grid loaded from {path.name} ({len(cached)} configs)")
            return cached
        print("[cache] grid.json does not match current actions, recomputing")
    grid = []
    for i, (a, n) in enumerate(configs, 1):
        t = run_trial(ctx, a, n)
        t["banned"] = bool(t["guardrail_applies"] and t["src_drop"] > CFG.guardrail_max_drop)
        grid.append(t)
        print(f"  grid {i:02d}/{len(configs)}  {a:17s} n={n:3d}  F1={t['f1_mean']:.3f}  "
              f"src_drop={t['src_drop']:.3f}  banned={t['banned']}")
    path.write_text(json.dumps(grid, indent=1, default=float), encoding="utf-8")
    return grid


def oracle(grid):
    """Same parsimony rule as Agent.select_final, applied to the full grid."""
    ok = [t for t in grid if not t["banned"]]
    top = max(ok, key=lambda t: t["f1_mean"])
    tied = [t for t in ok if paired_diff_ci(t["f1_per_seed"], top["f1_per_seed"])[2] >= 0]
    chosen = min(tied, key=lambda t: (t["n_labels"], SCREEN_ORDER.index(t["action"])))
    return top, chosen, tied


# ---------- agent (read from log) ----------
def load_agent():
    log, summ = LOG_DIR / "trials.jsonl", LOG_DIR / "agent_summary.json"
    if not log.exists() or not summ.exists():
        sys.exit("logs/trials.jsonl or agent_summary.json missing. Run: python src/agent.py")
    trials = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("type") == "trial":
                trials.append(e)
    return trials, json.loads(summ.read_text(encoding="utf-8"))


# ---------- random search simulation ----------
def simulate_random(grid, tied_keys, budget):
    """Random search with no guardrail knowledge: it spends trials on banned configs too.
    It picks the best-mean non-banned config among the trials it ran."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(grid)
    f1 = np.array([t["f1_mean"] for t in grid])
    banned = np.array([t["banned"] for t in grid])
    labels = np.array([t["n_labels"] for t in grid])
    secs = np.array([compute_sec(t) for t in grid])
    is_tied = np.array([key(t) in tied_keys for t in grid])

    best_f1, best_labels, used_secs, first_hit = [], [], [], []
    curves = np.full((N_RANDOM, n), np.nan)
    for r in range(N_RANDOM):
        order = rng.permutation(n)
        masked = np.where(banned[order], -np.inf, f1[order])
        running = np.maximum.accumulate(masked)
        curves[r] = np.where(np.isfinite(running), running, np.nan)
        head = masked[:budget]
        if np.isfinite(head.max()):
            j = int(np.argmax(head))
            best_f1.append(float(head[j]))
            best_labels.append(int(labels[order[j]]))
        else:
            best_f1.append(float("nan"))
            best_labels.append(0)
        used_secs.append(float(secs[order[:budget]].sum()))
        first_hit.append(int(np.argmax(is_tied[order])) + 1)   # trials until a tied config is tried

    best_f1 = np.array(best_f1)
    first_hit = np.array(first_hit)
    return {
        "budget": budget, "n_simulations": N_RANDOM,
        "best_f1_mean": float(np.nanmean(best_f1)),
        "best_f1_median": float(np.nanmedian(best_f1)),
        "best_f1_p5": float(np.nanpercentile(best_f1, 5)),
        "mean_labels_of_selected": float(np.mean(best_labels)),
        "mean_compute_sec": float(np.mean(used_secs)),
        "prob_reach_tied_set_within_budget": float(np.mean(first_hit <= budget)),
        "expected_trials_to_reach_tied_set": float(first_hit.mean()),
        "curve_mean_best_so_far": [float(x) for x in np.nanmean(curves, axis=0)],
    }


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    RESULTS_DIR.mkdir(exist_ok=True)
    ctx = Context()
    grid = build_grid(ctx, force="--force" in sys.argv)
    top, chosen, tied = oracle(grid)
    tied_keys = {key(t) for t in tied}

    # Agent
    a_trials, a_sum = load_agent()
    budget = a_sum["n_trials"]
    running, a_curve, a_hit = -np.inf, [], None
    for t in a_trials:
        if not t["banned"]:
            running = max(running, t["f1_mean"])
            if a_hit is None and key(t) in tied_keys:
                a_hit = t["trial_id"]
        a_curve.append(float(running) if np.isfinite(running) else None)
    a_secs = sum(compute_sec(t) for t in a_trials)
    a_sel = a_sum["selected"]

    # Random + fixed
    rnd = simulate_random(grid, tied_keys, budget)
    fixed = next(t for t in grid if key(t) == FIXED)

    # TF-IDF reference (baseline, NOT in the action menu)
    tfidf_ref = {}
    bpath = RESULTS_DIR / "baselines.json"
    if bpath.exists():
        tfidf_ref = json.loads(bpath.read_text(encoding="utf-8"))["few_shot"]["tfidf"]

    # ---------- report ----------
    print("\n=== Search comparison (YouTube VALIDATION Macro-F1, 5 seeds) ===")
    print(f"Oracle top mean : {fmt(top)}  F1={top['f1_mean']:.3f}")
    print(f"Oracle selected : {fmt(chosen)}  F1={chosen['f1_mean']:.3f}  (parsimony rule)")
    print(f"Not significantly worse than top: {[fmt(t) for t in tied]}")
    print(f"Banned configs in grid: {sum(t['banned'] for t in grid)} of {len(grid)}\n")

    print(f"{'method':22s} {'trials':>6s} {'best F1':>8s} {'labels':>7s} {'compute_s':>9s}  reached tied-set")
    print("-" * 82)
    print(f"{'Oracle (grid)':22s} {len(grid):>6d} {top['f1_mean']:>8.3f} {chosen['n_labels']:>7d} "
          f"{sum(compute_sec(t) for t in grid):>9.2f}  -")
    print(f"{'Agent':22s} {budget:>6d} {a_sum['top_mean']['f1_mean']:>8.3f} {a_sel['n_labels']:>7d} "
          f"{a_secs:>9.2f}  at trial {a_hit}  (stop: {a_sum['stop_reason']})")
    print(f"{'Random (mean of 500)':22s} {budget:>6d} {rnd['best_f1_mean']:>8.3f} "
          f"{rnd['mean_labels_of_selected']:>7.0f} {rnd['mean_compute_sec']:>9.2f}  "
          f"{rnd['prob_reach_tied_set_within_budget'] * 100:.0f}% of runs within budget; "
          f"expected {rnd['expected_trials_to_reach_tied_set']:.1f} trials")
    print(f"{'  random p5 (bad luck)':22s} {'':>6s} {rnd['best_f1_p5']:>8.3f}")
    print(f"{'Fixed recipe':22s} {1:>6d} {fixed['f1_mean']:>8.3f} {fixed['n_labels']:>7d} "
          f"{compute_sec(fixed):>9.2f}  {'yes' if FIXED in tied_keys else 'no'}")

    if tfidf_ref:
        print("\nReference, TF-IDF few-shot (baseline, not in action menu): " +
              "  ".join(f"{n}: {v[0]:.3f}" for n, v in tfidf_ref.items() if int(n) in LADDER))

    # What the agent never ran
    ran = {key(t) for t in a_trials}
    skipped = [t for t in grid if key(t) not in ran and not t["banned"]]
    best_skipped = max(skipped, key=lambda t: t["f1_mean"]) if skipped else None
    print(f"\nAgent ran {len(ran)} of {len(grid)} configs. "
          f"Best non-banned config it never ran: "
          f"{fmt(best_skipped) + f' F1={best_skipped['f1_mean']:.3f}' if best_skipped else 'none'}")

    out = {
        "oracle": {"top": fmt(top), "top_f1": top["f1_mean"], "selected": fmt(chosen),
                   "tied_with_top": [fmt(t) for t in tied], "n_configs": len(grid)},
        "agent": {"trials": budget, "stop_reason": a_sum["stop_reason"], "selected": a_sel,
                  "top_mean": a_sum["top_mean"], "reached_tied_set_at_trial": a_hit,
                  "compute_sec": a_secs, "curve_best_so_far": a_curve},
        "random": rnd,
        "fixed": {"config": fmt(fixed), "f1_mean": fixed["f1_mean"], "f1_ci": fixed["f1_ci"]},
        "tfidf_reference": tfidf_ref,
        "note": "Validation only. Winner's-curse caveat: best-of-many selection on a 200-example "
                "val set is optimistic; the test set gives the unbiased number.",
    }
    (RESULTS_DIR / "compare.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {RESULTS_DIR / 'compare.json'} and grid.json")


if __name__ == "__main__":
    main()