"""Print full text of the 10 hardest validation comments (selected model). Validation only."""
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from actions import ACTIONS, SEEDS, Context, _fit
from evaluate import sample_idx

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ctx = Context()
wrong = []
for s in SEEDS:
    idx = sample_idx(ctx.yp, 100, s)
    clf, feats = _fit(ctx, ACTIONS["hybrid_C10"], idx, s)
    wrong.append(clf.predict(feats(ctx.val_texts, ctx.Xv)) != ctx.yv)
rate = np.mean(wrong, axis=0)
for i in np.argsort(-rate)[:10]:
    print(f"[true={'spam' if ctx.yv[i] else 'ham'}] err={rate[i]:.1f}\n{ctx.val_texts[i]}\n")