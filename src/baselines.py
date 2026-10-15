"""Baselines: majority, TF-IDF, and MiniLM zero-shot / few-shot. Uses VALIDATION sets only."""
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from data import load_all
from embed import load_embeddings
from evaluate import evaluate_clf, mean_ci, sample_idx

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
LABEL_LADDER = [25, 50, 100, 200]


def tfidf_pipeline():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=1000),
    )


def main():
    texts = load_all()
    emb = load_embeddings()
    Xs_tr, ys_tr = emb["sms_train"]
    Xs_va, ys_va = emb["sms_val"]
    Xp, yp = emb["yt_pool"]
    Xv, yv = emb["yt_val"]
    results = {}

    # 1. Majority class: the floor
    maj = DummyClassifier(strategy="most_frequent").fit(Xp, yp)
    results["majority"] = evaluate_clf(maj, Xv, yv)

    # 2. TF-IDF trained on SMS, applied to YouTube (domain shift, no pretrained model)
    tf = tfidf_pipeline().fit(texts["sms_train"]["text"].tolist(), ys_tr)
    results["tfidf_zero_shot"] = evaluate_clf(tf, texts["yt_val"]["text"].tolist(), yv)

    # 3. A1 zero-shot: MiniLM embeddings + LogReg trained on SMS only, applied to YouTube
    src_clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_tr, ys_tr)
    results["minilm_zero_shot"] = {
        "target": evaluate_clf(src_clf, Xv, yv),
        "source_val": evaluate_clf(src_clf, Xs_va, ys_va),
    }

    # 4. Few-shot with target labels: TF-IDF vs frozen MiniLM + LogReg, 5 seeds each
    few = {"tfidf": {}, "minilm_logreg": {}}
    pool_texts = texts["yt_pool"]["text"].tolist()
    val_texts = texts["yt_val"]["text"].tolist()
    for n in LABEL_LADDER:
        tf_f1, ml_f1 = [], []
        for seed in SEEDS:
            idx = sample_idx(yp, n, seed)
            t = tfidf_pipeline().fit([pool_texts[i] for i in idx], yp[idx])
            tf_f1.append(evaluate_clf(t, val_texts, yv)["macro_f1"])
            m = LogisticRegression(max_iter=1000).fit(Xp[idx], yp[idx])
            ml_f1.append(evaluate_clf(m, Xv, yv)["macro_f1"])
        few["tfidf"][n] = mean_ci(tf_f1)
        few["minilm_logreg"][n] = mean_ci(ml_f1)
    results["few_shot"] = few

    # Print summary
    print("=== Baselines (YouTube VALIDATION, Macro-F1) ===")
    print(f"majority class        : {results['majority']['macro_f1']:.3f}")
    print(f"TF-IDF  SMS->YouTube  : {results['tfidf_zero_shot']['macro_f1']:.3f}")
    zs = results["minilm_zero_shot"]
    print(f"MiniLM  SMS->YouTube  : {zs['target']['macro_f1']:.3f}   (zero-shot, A1)")
    print(f"MiniLM  on SMS val    : {zs['source_val']['macro_f1']:.3f}   (source F1, forgetting reference)")
    print("\nFew-shot with target labels (mean [95% CI] over 5 seeds):")
    print(f"{'labels':>6} | {'TF-IDF':^24} | {'MiniLM+LogReg':^24}")
    for n in LABEL_LADDER:
        a, b = few["tfidf"][n], few["minilm_logreg"][n]
        print(f"{n:>6} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_DIR / 'baselines.json'}")


if __name__ == "__main__":
    main()