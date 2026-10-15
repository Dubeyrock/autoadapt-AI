"""Encoder + embedding cache: encode each split once with a frozen MiniLM and save to cache/."""
import time
from importlib import import_module
from pathlib import Path

import numpy as np

from data import load_all

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_encoder():
    try:
        return import_module("sentence_transformers").SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers and its dependencies must be installed in the "
            "active environment. Run: python -m pip install -r requirements.txt"
        ) from exc


def cache_path(split_name):
    return CACHE_DIR / f"{split_name}.npy"


def build_cache(force=False, batch_size=64):
    """Encode every split once. Skips splits that are already cached unless force=True."""
    CACHE_DIR.mkdir(exist_ok=True)
    splits = load_all()
    model = None

    for name, df in splits.items():
        path = cache_path(name)
        if path.exists() and not force:
            print(f"[skip] {name}: already cached")
            continue
        if model is None:
            print(f"Loading encoder: {MODEL_NAME}")
            model = _load_encoder()(MODEL_NAME, device="cpu")
        t0 = time.time()
        X = model.encode(
            df["text"].tolist(), batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        ).astype(np.float32)
        np.save(path, X)
        print(f"[done] {name}: {X.shape} in {time.time() - t0:.1f}s")


def load_embeddings():
    """Return (X, y) per split from the cache. Raises if the cache is missing."""
    splits = load_all()
    out = {}
    for name, df in splits.items():
        path = cache_path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python src/embed.py")
        X = np.load(path)
        assert len(X) == len(df), f"{name}: cache has {len(X)} rows but split has {len(df)}"
        out[name] = (X, df["y"].to_numpy())
    return out


def measure_encode_latency(n=100):
    """Single-text encoding latency in ms (for the latency metric in the report)."""
    splits = load_all()
    texts = splits["yt_val"]["text"].tolist()[:n]
    model = _load_encoder()(MODEL_NAME, device="cpu")
    model.encode(texts[:5])  # warm-up
    t0 = time.time()
    for t in texts:
        model.encode([t])
    return (time.time() - t0) / len(texts) * 1000


if __name__ == "__main__":
    build_cache()
    data = load_embeddings()
    print()
    for name, (X, y) in data.items():
        print(f"{name:10s} X={X.shape}  y={y.shape}  spam_ratio={y.mean():.3f}")
    print(f"\nEncode latency (single text, CPU): {measure_encode_latency():.1f} ms")