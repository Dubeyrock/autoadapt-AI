"""Adaptation actions. Each action takes (context, n_labels, seed) and returns a metrics dict.
Uses VALIDATION sets only. The test set is never touched here."""
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

import time

from data import load_all
from embed import load_embeddings
from evaluate import evaluate_clf, mean_ci, sample_idx

SEEDS = [0, 1, 2, 3, 4]

# name -> spec. Hyperparameters are untuned starting guesses (the agent compares them).
ACTIONS = {
    "zero_shot":        dict(kind="zero_shot"),
    "logreg_C1":        dict(kind="logreg", C=1.0),
    "logreg_C10":       dict(kind="logreg", C=10.0),
    "logreg_C100":      dict(kind="logreg", C=100.0),
    "svm_C1":           dict(kind="svm", C=1.0),
    "mlp":              dict(kind="mlp"),
    "warmstart_gentle": dict(kind="warmstart", lr=0.1, epochs=10, guardrail=True),
    "warmstart_strong": dict(kind="warmstart", lr=1.0, epochs=50, guardrail=True),
    "hybrid_C10":       dict(kind="hybrid", C=10.0),
}


class Context:
    """Holds cached embeddings, texts, and the source (SMS) classifier."""

    def __init__(self):
        texts = load_all()
        emb = load_embeddings()
        self.Xs_tr, self.ys_tr = emb["sms_train"]
        self.Xs_va, self.ys_va = emb["sms_val"]
        self.Xp, self.yp = emb["yt_pool"]
        self.Xv, self.yv = emb["yt_val"]
        self.pool_texts = texts["yt_pool"]["text"].tolist()
        self.val_texts = texts["yt_val"]["text"].tolist()
        self.sms_val_texts = texts["sms_val"]["text"].tolist()
        self.src_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(self.Xs_tr, self.ys_tr)
        self.src_f1 = evaluate_clf(self.src_clf, self.Xs_va, self.ys_va)["macro_f1"]


def _identity(texts, X):
    return X


def _fit(ctx, spec, idx, seed):
    """Train a head on the labeled subset. Returns (classifier, feature_fn)."""
    kind = spec["kind"]
    Xt, yt = ctx.Xp[idx], ctx.yp[idx]

    if kind == "logreg":
        return LogisticRegression(C=spec["C"], max_iter=2000).fit(Xt, yt), _identity
    if kind == "svm":
        return LinearSVC(C=spec["C"], max_iter=5000).fit(Xt, yt), _identity
    if kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, random_state=seed)
        return clf.fit(Xt, yt), _identity
    if kind == "warmstart":
        # Start from the SMS classifier's weights, then a limited number of SGD steps on target labels.
        clf = SGDClassifier(
            loss="log_loss", learning_rate="constant", eta0=spec["lr"], alpha=1e-4,
            max_iter=spec["epochs"], tol=None, random_state=seed,
        )
        clf.fit(Xt, yt, coef_init=ctx.src_clf.coef_.copy(), intercept_init=ctx.src_clf.intercept_.copy())
        return clf, _identity
    if kind == "hybrid":
        sub_texts = [ctx.pool_texts[i] for i in idx]
        tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(sub_texts)

        def feats(texts, X):
            return sp.hstack([sp.csr_matrix(X), tf.transform(texts)]).tocsr()

        clf = LogisticRegression(C=spec["C"], max_iter=2000).fit(feats(sub_texts, Xt), yt)
        return clf, feats
    raise ValueError(f"unknown action kind: {kind}")


def run_action(ctx, name, n_labels, seed):
    spec = ACTIONS[name]
    t0 = time.perf_counter()
    if spec["kind"] == "zero_shot":
        clf, feats, n_used = ctx.src_clf, _identity, 0
    else:
        idx = sample_idx(ctx.yp, n_labels, seed)
        clf, feats = _fit(ctx, spec, idx, seed)
        n_used = len(idx)
    train_sec = time.perf_counter() - t0

    tgt = evaluate_clf(clf, feats(ctx.val_texts, ctx.Xv), ctx.yv)
    src = evaluate_clf(clf, feats(ctx.sms_val_texts, ctx.Xs_va), ctx.ys_va)
    return {
        "action": name, "n_labels": n_used, "seed": seed,
        "macro_f1": tgt["macro_f1"], "precision": tgt["precision"], "recall": tgt["recall"],
        "infer_ms": tgt["infer_ms"],          # head only; add encoder latency for end-to-end
        "src_f1": src["macro_f1"], "src_drop": ctx.src_f1 - src["macro_f1"],
        "train_sec": train_sec,
        "guardrail_applies": bool(spec.get("guardrail", False)),
    }


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    ctx = Context()
    print(f"Source reference (SMS val Macro-F1): {ctx.src_f1:.3f}\n")
    print(f"{'action':18s} {'n':>4s} | {'target F1 mean [95% CI]':^26s} | {'src_drop':>8s} | {'train_s':>7s}")
    print("-" * 74)
    for name in ACTIONS:
        ns = [0] if name == "zero_shot" else [50, 200]
        for n in ns:
            seeds = [0] if n == 0 else SEEDS
            rows = [run_action(ctx, name, n, s) for s in seeds]
            m, lo, hi = mean_ci([r["macro_f1"] for r in rows])
            drop = np.mean([r["src_drop"] for r in rows])
            sec = np.mean([r["train_sec"] for r in rows])
            print(f"{name:18s} {n:>4d} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {drop:>8.3f} | {sec:>7.3f}")