"""Final evaluation on the TEST set. Run this ONCE, after the agent has selected a model.
Refuses to run again unless --force is passed. Nothing here feeds back into model selection."""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix

from actions import ACTIONS, SEEDS, Context, _fit
from baselines import tfidf_pipeline
from data import load_all
from embed import load_embeddings
from evaluate import evaluate_clf, mean_ci, sample_idx
from stop_rules import paired_diff_ci

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "final_test.json"


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT.name} already exists. The test set is used once. "
                 "Re-running would break that rule (use --force only if you log it in Deviations).")

    summ = json.loads((ROOT / "logs" / "agent_summary.json").read_text(encoding="utf-8"))
    name, n = summ["selected"]["action"], summ["selected"]["n_labels"]
    val_f1 = summ["selected"]["f1_mean"]
    print(f"Selected by agent (validation only): {name}@{n}  val F1={val_f1:.3f}")

    ctx = Context()
    texts = load_all()
    Xt, yt = load_embeddings()["yt_test"]
    test_texts = texts["yt_test"]["text"].tolist()

    sel, tfidf, fixed = [], [], []
    cm = np.zeros((2, 2), dtype=int)
    for seed in SEEDS:
        idx = sample_idx(ctx.yp, n, seed)
        clf, feats = _fit(ctx, ACTIONS[name], idx, seed)
        sel.append(evaluate_clf(clf, feats(test_texts, Xt), yt)["macro_f1"])
        cm += confusion_matrix(yt, clf.predict(feats(test_texts, Xt)), labels=[0, 1])

        tf = tfidf_pipeline().fit([ctx.pool_texts[i] for i in idx], ctx.yp[idx])
        tfidf.append(evaluate_clf(tf, test_texts, yt)["macro_f1"])

        idx_f = sample_idx(ctx.yp, 200, seed)
        fc, ff = _fit(ctx, ACTIONS["logreg_C1"], idx_f, seed)
        fixed.append(evaluate_clf(fc, ff(test_texts, Xt), yt)["macro_f1"])

    maj = evaluate_clf(DummyClassifier(strategy="most_frequent").fit(ctx.Xp, ctx.yp), Xt, yt)["macro_f1"]
    zs = evaluate_clf(ctx.src_clf, Xt, yt)["macro_f1"]
    d_tf = paired_diff_ci(sel, tfidf)
    d_fx = paired_diff_ci(sel, fixed)
    m, lo, hi = mean_ci(sel)

    print(f"\n=== TEST (YouTube, n={len(yt)}, Macro-F1, 5 label-subsets) ===")
    print(f"Selected {name}@{n}   : {m:.3f} [{lo:.3f}, {hi:.3f}]   (val was {val_f1:.3f}, gap {val_f1 - m:+.3f})")
    print(f"TF-IDF@{n} (reference): {np.mean(tfidf):.3f}")
    print(f"Fixed logreg_C1@200   : {np.mean(fixed):.3f}")
    print(f"Zero-shot             : {zs:.3f}")
    print(f"Majority              : {maj:.3f}")
    print(f"Selected - TF-IDF  paired diff: {d_tf[0]:+.3f} [{d_tf[1]:+.3f}, {d_tf[2]:+.3f}]")
    print(f"Selected - Fixed   paired diff: {d_fx[0]:+.3f} [{d_fx[1]:+.3f}, {d_fx[2]:+.3f}]")
    print(f"Confusion matrix summed over seeds [[TN, FP], [FN, TP]]:\n{cm}")
    print("Note: test n=300, so ~±3 points of sampling noise remains.")

    OUT.write_text(json.dumps({
        "selected": f"{name}@{n}", "val_f1": val_f1, "test_f1": [m, lo, hi],
        "tfidf_test_f1": float(np.mean(tfidf)), "fixed_test_f1": float(np.mean(fixed)),
        "zero_shot_test_f1": zs, "majority_test_f1": maj,
        "diff_vs_tfidf": d_tf, "diff_vs_fixed": d_fx, "confusion_matrix": cm.tolist(),
        "per_seed_selected": sel,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()