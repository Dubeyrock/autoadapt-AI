"""Data layer: load, dedupe, and split the SMS (source) and YouTube (target) datasets."""
import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

YT_TEST_SIZE = 300
YT_VAL_SIZE = 200


def load_sms(path=None, verbose=False):
    path = Path(path) if path else DATA_DIR / "SMSSpamCollection"
    df = pd.read_csv(
        path, sep="\t", header=None, names=["label", "text"],
        quoting=csv.QUOTE_NONE, encoding="utf-8", encoding_errors="replace",
    )
    df["y"] = (df["label"] == "spam").astype(int)
    df = df[["text", "y"]].dropna()
    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if verbose:
        print(f"[SMS] raw={n_raw}, duplicates removed={n_raw - len(df)}, final={len(df)}")
    return df


def load_youtube(data_dir=None, verbose=False):
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    files = sorted(data_dir.glob("Youtube*.csv"))
    assert files, f"YouTube CSV files not found in {data_dir}"
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.rename(columns={"CONTENT": "text", "CLASS": "y"})
    df = df[["text", "y"]].dropna()
    df["y"] = df["y"].astype(int)
    n_raw = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    if verbose:
        print(f"[YouTube] files={len(files)}, raw={n_raw}, duplicates removed={n_raw - len(df)}, final={len(df)}")
    return df


def split_sms(df):
    """70/15/15 stratified."""
    train, rest = train_test_split(df, test_size=0.30, stratify=df["y"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["y"], random_state=SEED)
    return (train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def split_youtube(df):
    """Test is carved out FIRST and must only be used once, on the final model.
    Then val; the remainder is the label pool for 25/50/100/200 samples."""
    rest, test = train_test_split(df, test_size=YT_TEST_SIZE, stratify=df["y"], random_state=SEED)
    pool, val = train_test_split(rest, test_size=YT_VAL_SIZE, stratify=rest["y"], random_state=SEED)
    return (pool.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True))


def load_all(verbose=False):
    sms = load_sms(verbose=verbose)
    yt = load_youtube(verbose=verbose)
    s_tr, s_va, s_te = split_sms(sms)
    y_pool, y_va, y_te = split_youtube(yt)
    return {
        "sms_train": s_tr, "sms_val": s_va, "sms_test": s_te,
        "yt_pool": y_pool, "yt_val": y_va, "yt_test": y_te,
    }


if __name__ == "__main__":
    splits = load_all(verbose=True)
    print()
    for name, d in splits.items():
        print(f"{name:10s} n={len(d):5d}  spam_ratio={d['y'].mean():.3f}")

    # Leakage check: no text should appear in more than one YouTube split
    yt_sets = {k: set(v["text"]) for k, v in splits.items() if k.startswith("yt_")}
    overlap = (yt_sets["yt_pool"] & yt_sets["yt_val"]) | (yt_sets["yt_pool"] & yt_sets["yt_test"]) | (yt_sets["yt_val"] & yt_sets["yt_test"])
    print("\nYouTube leakage check, overlapping texts across splits:", len(overlap))