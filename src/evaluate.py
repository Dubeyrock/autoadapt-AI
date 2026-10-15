"""Evaluator: metrics, latency, confidence intervals, and stratified label sampling."""
import time

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def sample_idx(y, n, seed):
    """Stratified sample of n indices from the label pool (both classes always present)."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return idx


def evaluate_clf(clf, X, y):
    """Macro-F1, spam precision/recall, and per-sample inference latency (ms)."""
    t0 = time.perf_counter()
    pred = clf.predict(X)
    infer_ms = (time.perf_counter() - t0) / len(y) * 1000
    return {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "infer_ms": float(infer_ms),
    }


def mean_ci(values, conf=0.95):
    """Mean and t-distribution confidence interval across seeds -> (mean, low, high)."""
    v = np.asarray(values, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, m, m
    h = float(stats.sem(v) * stats.t.ppf((1 + conf) / 2, len(v) - 1))
    return m, m - h, m + h