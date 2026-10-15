"""Error analysis on VALIDATION predictions only. Compares the selected model, a MiniLM-only head
and TF-IDF over the same 5 label subsets. Tests hypothesis H1 (see README Deviations log):
MiniLM-only misses more spam that carries lexical cues than TF-IDF does. Descriptive only."""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score

from actions import ACTIONS, SEEDS, Context, _fit
from baselines import tfidf_pipeline
from evaluate import sample_idx

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "error_analysis.json"

# Cue definition fixed here, before looking at any error analysis output.
CUE = re.compile(r"https?:|www\.|\.com|\.net|subscribe|check (?:out|my|this)|channel|follow me|visit|click", re.I)


def predict_all(ctx, name, n):
    """Validation predictions for every seed -> array (n_seeds, n_val)."""
    preds = []
    for seed in SEEDS:
        idx = sample_idx(ctx.yp, n, seed)
        if name == "tfidf":
            m = tfidf_pipeline().fit([ctx.pool_texts[i] for i in idx], ctx.yp[idx])
            preds.append(m.predict(ctx.val_texts))
        else:
            clf, feats = _fit(ctx, ACTIONS[name], idx, seed)
            preds.append(clf.predict(feats(ctx.val_texts, ctx.Xv)))
    return np.array(preds)


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console + odd characters

    summ = json.loads((ROOT / "logs" / "agent_summary.json").read_text(encoding="utf-8"))
    sel, n = summ["selected"]["action"], summ["selected"]["n_labels"]
    ctx = Context()
    y, texts = ctx.yv, ctx.val_texts

    models = {f"{sel} (selected)": sel, "logreg_C10 (MiniLM only)": "logreg_C10", "tfidf": "tfidf"}
    preds = {label: predict_all(ctx, name, n) for label, name in models.items()}
    wrong = {label: p != y for label, p in preds.items()}          # (seeds, n_val)
    err_rate = {label: w.mean(axis=0) for label, w in wrong.items()}

    print(f"=== Error analysis, YouTube VALIDATION (n={len(y)}), {n} labels, {len(SEEDS)} label subsets ===\n")
    report = {"n_labels": n, "models": {}}
    for label, p in preds.items():
        f1 = float(np.mean([f1_score(y, row, average="macro") for row in p]))
        cm = sum(confusion_matrix(y, row, labels=[0, 1]) for row in p)
        print(f"{label:26s} Macro-F1={f1:.3f}   [[TN, FP], [FN, TP]] summed over seeds = {cm.tolist()}")
        report["models"][label] = {"macro_f1": f1, "confusion_matrix": cm.tolist()}

    # H1: spam recall with vs without lexical cues
    cue = np.array([bool(CUE.search(t)) for t in texts])
    spam = y == 1
    groups = {"spam WITH cue": spam & cue, "spam WITHOUT cue": spam & ~cue}
    print("\n--- H1: spam recall by lexical-cue group (mean over seeds) ---")
    print(f"{'group':18s} {'n':>4s} | " + " | ".join(f"{m[:24]:>24s}" for m in preds))
    report["h1"] = {}
    for g, mask in groups.items():
        rec = {m: float(1 - err_rate[m][mask].mean()) for m in preds}
        print(f"{g:18s} {int(mask.sum()):>4d} | " + " | ".join(f"{rec[m]:>24.3f}" for m in preds))
        report["h1"][g] = {"n": int(mask.sum()), "recall": rec}
    print("Groups are small (validation n=200), so this is descriptive. No significance claimed.")

    # Hardest examples for the selected model
    sel_label = f"{sel} (selected)"
    order = np.argsort(-err_rate[sel_label])
    hard = [i for i in order if err_rate[sel_label][i] > 0][:10]
    print(f"\n--- 10 hardest validation comments for {sel_label} (error rate over {len(SEEDS)} seeds) ---")
    report["hardest"] = []
    for i in hard:
        rates = {m: float(err_rate[m][i]) for m in preds}
        txt = " ".join(str(texts[i]).split())[:90]
        print(f"[true={'spam' if y[i] else 'ham '}] err sel={rates[sel_label]:.1f} "
              f"minilm={rates['logreg_C10 (MiniLM only)']:.1f} tfidf={rates['tfidf']:.1f} | {txt}")
        report["hardest"].append({"text": txt, "true": int(y[i]), "err_rate": rates})

    both = int(np.sum((err_rate[sel_label] >= 0.6) & (err_rate["tfidf"] >= 0.6)))
    sel_only = int(np.sum(err_rate[sel_label] >= 0.6))
    print(f"\nConsistent errors (wrong in >=3/5 seeds): selected={sel_only}, also wrong for TF-IDF={both}")
    report["consistent_errors"] = {"selected": sel_only, "shared_with_tfidf": both}

    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()