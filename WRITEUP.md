# AutoAdapt: Writeup

## 1. Framing

The brief: adapt a pretrained model to a new task without its original training data, then build an
agent that manages the adaptation and knows when to stop. I read two constraints in that sentence:
(1) the original pretraining corpus is unavailable, so no replay and no retraining from scratch;
(2) the stopping decision is a deliverable, not a footnote.

**Choices**
- Model: `all-MiniLM-L6-v2`, kept **frozen**. Its pretrained knowledge cannot be "forgotten" by construction.
  The only forgetting I can measure is on an SMS head, which is a *proxy*, not the encoder's pretraining data.
- Task: spam detection. Source = SMS Spam (5,171 messages after removing 403 duplicates).
  Target = YouTube comments (1,760 after removing 196 duplicates). Real domain shift: an SMS-trained
  classifier scores 0.959 Macro-F1 on SMS but only 0.548 on YouTube.
- Splits (seed 42): YouTube test 300 (evaluated once), val 200, label pool 1,260. SMS 70/15/15.
- Success criteria (metric, 5 seeds, guardrail, budget, stop rules, test-set-once) were written in the README
  before any experiment. Later changes are in its Deviations log with reasons.

**Deliberately not built:** LoRA / partial fine-tuning (A6), an LLM-driven agent, a second target domain,
and pseudo-labelling. A small read-only Streamlit dashboard was added as a presentation layer after the
experiments; it displays saved artifacts and is not part of model selection or evaluation. With frozen
heads every config trains in well under a second, so the full 25-config space can be run exhaustively and
used as an oracle to audit the agent.

## 2. What I built

```
data.py -> embed.py (frozen MiniLM, cached .npy) -> actions.py (9 adaptation actions)
        -> evaluate.py (Macro-F1, paired CI, latency) -> agent.py + stop_rules.py -> logs/trials.jsonl
        -> compare.py / plots.py / final_eval.py / error_analysis.py
        -> app.py (read-only Streamlit view of saved evidence)
```

- **Actions:** zero-shot, LogReg (C=1/10/100), LinearSVM, MLP head, two warm-start variants (start from the
  SMS classifier's weights), hybrid (MiniLM embeddings + TF-IDF features). Hyperparameters are untuned guesses.
- **Agent (rule-based, no LLM):** screen every action once at 50 labels, cheapest first; ban warm-start
  actions that break the guardrail (SMS Macro-F1 drop > 3 points); escalate the top 2 to 100 then 200 labels;
  freeze an action when its gain per extra label falls below 0.0002; check the pruned actions once at 200;
  stop on plateau / budget / all-banned. Every decision and stop reason is logged. "Better" always means
  a paired 95% CI over 5 label-subset seeds that excludes zero.
- **Final selection (parsimony):** among configs not significantly worse than the top one, use the fewest labels.
- Selection used validation data only. Final evaluation ran once after selection; its output is saved as
  `results/final_test.json`. `final_eval.py` blocks a repeat in the same project copy. A new copy without
  that artifact cannot know the test set was already used, so preserve this file and do not run
  `final_eval.py` during fresh-environment verification.
- Tests: 15 pytest tests (split leakage, determinism, each stop rule on synthetic histories, guardrail,
  agent stopping on a small trial budget, selection code never references the test splits).
- Encoder latency (single text, CPU): **37.7 ms** in the latest recorded run. This is machine- and load-dependent;
  head-only inference latency is logged per trial.

## 3. Results

**Baselines (YouTube val Macro-F1)**

| Method | Score |
|---|---|
| Majority | 0.342 |
| TF-IDF trained on SMS | 0.365 |
| MiniLM + LogReg trained on SMS (zero-shot) | 0.548 |

**Few-shot, TF-IDF vs frozen MiniLM + LogReg(C=1), mean over 5 seeds**

| Labels | TF-IDF | MiniLM + LogReg |
|---|---|---|
| 25 | 0.836 | 0.802 |
| 50 | 0.883 | 0.850 |
| 100 | 0.922 | 0.871 |
| 200 | 0.940 | 0.879 |

**Search comparison (validation, 25-config space)**

| Method | Trials | Best F1 | Labels of selected | Compute |
|---|---|---|---|---|
| Oracle (full grid) | 25 | 0.935 (selected 0.920) | 100 | 16.2 s |
| Agent | 13 (stopped: plateau) | 0.935 (selected 0.920) | 100 | 8.5 s |
| Random search (mean of 500) | 13 | 0.923 | ~169 | 8.5 s |
| Fixed recipe (LogReg C=1 @200) | 1 | 0.879 | 200 | 0.1 s |

**Test set (n=300, used once)**

| Model | Macro-F1 |
|---|---|
| Selected `hybrid_C10@100` | **0.927** [0.916, 0.939] (val was 0.920) |
| TF-IDF @100 | 0.926 |
| Fixed recipe `logreg_C1@200` | 0.911 |
| Zero-shot | 0.551 |
| Majority | 0.344 |

Paired differences on test: selected vs fixed +0.016 [+0.006, +0.027]; selected vs TF-IDF +0.001 [-0.023, +0.025].
Plots: `results/plot1_f1_vs_labels.png`, `plot2_tradeoff.png`, `plot3_search.png`. Evidence: `logs/trials.jsonl`
(every trial, decision and stop reason), `logs/agent_summary.json`, `results/*.json`.

## 4. What failed or did not hold

1. **The agent did not find the answer faster than random search.** It matched the oracle's choice
   (`hybrid_C10@100`) with 13 of 25 trials and about half the compute of the grid, but random search reaches a
   config in the top tied set in an expected 8.4 trials versus 10 for the agent. Compute is seconds either way,
   so the real cost is labels. What the agent gives is determinism (random's 5th-percentile best F1 is 0.895),
   no wasted trials on guardrail-breaking configs, a self-chosen stop point, and fewer labels. The label
   comparison is slightly unfair to random: it has no parsimony rule.
2. **Both warm-start actions were banned.** Source drop was 0.14 to 0.18 (gentle) and 0.31 to 0.37 (strong)
   against a 0.03 limit. Gentle forgets less but adapts poorly (0.761 at 50 labels). The final model is a
   target-only head with a source drop of about 0.40, so it is domain-specific and does not serve both
   domains. I did not solve forgetting. The guardrail removed every one-model-for-both option.
3. **Frozen MiniLM added no measurable value over TF-IDF on this target.** Val: TF-IDF beats MiniLM-only at
   50 to 200 labels. Test: hybrid vs TF-IDF is +0.001 with a CI of about +/-2.5 points. I cannot say TF-IDF
   is better, only that I found no benefit from the encoder.
4. **My hypothesis H1 was refuted.** I predicted MiniLM-only misses lexical-cue spam (URLs, "subscribe",
   "check out") more than TF-IDF does. Recall of MiniLM-only was below TF-IDF in both groups: with cue 0.884
   vs 0.954 (n=79), without cue 0.447 vs 0.565 (n=17). The gap is not cue-specific. The no-cue group is tiny.
   I do not know the mechanism.
5. **Hardest validation errors (n=10, my reading, labels unverified):** 4 engagement-bait comments with no link
   ("give it a like"), 2 half-benign half-promo comments, 1 clear link spam missed by hybrid but caught by
   TF-IDF, 1 YouTube share-button boilerplate, 2 with questionable labels. 11 of the selected model's 14
   consistent errors are also wrong for TF-IDF, so part of the remaining error looks like task ambiguity.
6. **Process:** the action menu was extended (C sweep, hybrid) after I saw baseline numbers, and the
   diminishing-returns threshold was set after seeing baseline curves. Both are in the Deviations log.
   The 25-config space is small, so "the agent matches the oracle" is a weak claim about agents in general.

## 5. Limitations

- Test n=300 and val n=200: about +/-3 points of sampling noise remains. My CIs capture label-subset noise only,
  not test-set sampling noise, so they look tighter than the truth.
- "Not significantly worse" with 5 seeds is low-powered. It does not mean "equal".
- One encoder, one task, one target domain. Nothing here shows the agent generalises.
- Forgetting is measured on SMS only, not on the encoder's pretraining data.
- Raw text was not cleaned (HTML entities like `&#39;` and `<br />` remain).
- Package versions are lower-bounded in `requirements.txt`; no lockfile is included, so exact dependency
  versions may differ in a fresh environment.

## 6. What I would do differently

- Put TF-IDF in the action menu from the start, since it turned out to be the strongest single method.
- Replace the single 200-example val set with cross-validation over pool+val to cut selection noise.
- Test the agent on a second, larger action space and a second target domain (e.g. Enron email), where
  random search and grid search stop being trivially cheap.
- Add partial fine-tuning or LoRA with a replay-free regulariser, since that is the only route to a model that
  could keep SMS accuracy.
- Tune the decision threshold (validation showed more missed spam than false alarms), and audit the labels of the hardest comments.

## 7. Reproduce

The saved `results/final_test.json` is the result of the one permitted test evaluation. Keep this file
when preparing a fresh environment. The following commands reproduce the data processing, validation
experiments, plots, and tests without running the final test evaluation again:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\data.py
python src\embed.py
python src\baselines.py
python src\agent.py
python src\compare.py
python src\plots.py
python -m pytest tests -q
```

These commands regenerate validation outputs and may change measured timings slightly. Do not run
`python src\final_eval.py` during this verification: copying source and data without the saved test
artifact does not preserve the test-once safeguard. The test result in this write-up is the recorded
evaluation in `results/final_test.json`.