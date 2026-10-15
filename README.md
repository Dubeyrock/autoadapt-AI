# AutoAdapt: An Autonomous Agent for Data-Efficient Model Adaptation

An agent that adapts a pretrained text encoder to a new domain using very few
labeled examples, compares adaptation strategies under a fixed budget, and
decides on its own when to stop.

> **Status:** experiments have run; a fresh-environment reproduction has not
> yet been verified. See the checklist at the bottom.

## 1. Problem framing

- **Pretrained model:** `sentence-transformers/all-MiniLM-L6-v2` (22M params, CPU-friendly).
- **Source domain:** SMS spam (UCI SMS Spam Collection).
- **Target domain:** YouTube comment spam (UCI YouTube Spam Collection).
- **Task:** binary spam detection. Adapt to the target domain with 25 / 50 / 100 / 200 labels.
- **No access to original training data:** MiniLM's pretraining corpus is not
  available to us. We never retrain the encoder from scratch. The encoder is kept
  frozen and only lightweight heads are adapted on top of its embeddings.
- **Forgetting proxy:** for strategies that start from the SMS classifier
  (warm-start), we measure the F1 drop on SMS validation data. This is a *proxy*
  for retention of the source task. It is **not** the encoder's original
  pretraining data, so claims about forgetting are limited accordingly.

## 2. Success criteria (fixed BEFORE running experiments)

| Item | Definition |
|---|---|
| Primary metric | Target (YouTube) validation **Macro-F1** |
| Secondary metrics | Precision, recall, training time, inference latency, number of target labels used |
| Guardrail | Source (SMS) validation F1 drop <= **3 points** vs. the source-only classifier. A strategy that breaks it is banned. |
| Noise handling | Every trial runs with **5 seeds**; we report mean and 95% confidence interval |
| Budget | At most **20 trials** and **10 minutes** total |
| Stop rule: plateau | Last **3** trials are not significantly better than the best so far (CI overlap) |
| Stop rule: diminishing returns | Macro-F1 gain per additional label falls below a preset threshold |
| Stop rule: guardrail | All remaining actions are banned by the guardrail |
| Stop rule: budget | Trial or time limit reached |
| Test set | Target test set is used **exactly once**, on the final selected model |
| Reproducibility | Fixed seeds, pinned split, one-command run |

Latency is measured and reported but is not a hard gate in v1.

If any criterion changes after we see results, the change goes into the
**Deviations log** below with a reason.

## 3. Architecture

```
DATA LAYER  ->  ENCODER + EMBEDDING CACHE  ->  ADAPTATION ACTIONS
                                                      |
        LOGS / PLOTS / REPORT  <-  AGENT CONTROLLER  <-  EVALUATOR
```

| Component | File | Output |
|---|---|---|
| Data layer (load, dedupe, splits) | `src/data.py` | in-memory splits |
| Encoder + embedding cache | `src/embed.py` | `cache/*.npy` |
| Adaptation actions (A1-A5) | `src/actions.py` | trained heads |
| Evaluator (F1, forgetting, latency, CI) | `src/evaluate.py` | metric dicts |
| Agent controller | `src/agent.py` | `logs/trials.jsonl` |
| Stop rules | `src/stop_rules.py` | stop reason in log |
| Baselines, comparison, plots | `src/compare.py`, `src/plots.py` | `results/` |

### Adaptation actions

| ID | Action | Description |
|---|---|---|
| A1 | Zero-shot | SMS-trained classifier applied directly to YouTube |
| A2 | LogReg | Logistic regression on frozen embeddings, target labels |
| A3 | LinearSVM | Linear SVM on frozen embeddings, target labels |
| A4 | MLP head | Small MLP on frozen embeddings |
| A5 | Warm-start | Start from the SMS classifier, then fit on target labels |

### Agent policy (rule-based, no LLM)

1. Run all cheap actions at 50 labels.
2. Escalate only the top 2 actions to 100, then 200 labels.
3. Drop actions that are within the CI of the baseline.
4. Apply guardrail and stop rules after every trial, and log which one fired.

### Baselines and comparison

- Majority class (floor)
- TF-IDF + Logistic Regression (non-pretrained reference)
- Zero-shot transfer (size of the domain shift)
- Agent vs. random search vs. fixed recipe (always LogReg at 200 labels), all
  under the same budget

## 4. Data

| Dataset | Source | Role |
|---|---|---|
| SMS Spam Collection | UCI ML Repository | Source domain |
| YouTube Spam Collection | UCI ML Repository | Target domain |

Raw files live in `data/` (see `data/readme` for the original description and
license; please verify license terms there before redistributing).

Splits (seed 42, stratified):
- SMS: 70 / 15 / 15 train / val / test
- YouTube: test (300) set aside first, then val (200), the rest is the label pool
  from which 25 / 50 / 100 / 200 examples are sampled
- Exact duplicate comments are removed before splitting to avoid leakage

**Citations**
- Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng.
- Alberto, T.C., Lochter, J.V., Almeida, T.A. (2015). *TubeSpam: Comment spam filtering on YouTube.* IEEE ICMLA.
- Reimers, N., Gurevych, I. (2019). *Sentence-BERT.* EMNLP. Model: `all-MiniLM-L6-v2`.

## 5. Setup and run

Python 3.10+ recommended.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

Run the full pipeline from the project root after activating the venv:

```bat
run_all.bat
```

For macOS/Linux or Git Bash, use `bash run_all.sh`. The first run downloads the MiniLM model (~90 MB);
embeddings are cached in `cache/`. The pipeline regenerates validation logs, metrics, plots, error
analysis, and test results. It skips held-out test evaluation when `results/final_test.json` exists;
keep that file with the submission. If it is missing, the script runs final evaluation once.

### Interview dashboard

Install the project dependencies (including Streamlit) and start the read-only dashboard from the
project root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard visualizes the saved agent log and JSON results. It does not train models or run final
test evaluation. Its **About** page is the landing page, followed by **Overview**, **Agent decisions**,
and **Results & caveats**. Keep `logs/` and `results/` beside `app.py` so the saved evidence can be loaded.

`python src/final_eval.py` evaluates the held-out YouTube test set and must be run only once, after
model selection. It saves `results/final_test.json` and refuses to run again in the same project copy.
Preserve that file when validating the project in a fresh environment, and do not run `final_eval.py`
again: a fresh copy without the saved result cannot detect that the test set was already used.

## 6. Repository layout

```
autoadapt-AI/
├── data/           raw datasets
├── cache/          cached embeddings (.npy)
├── src/            all code
├── logs/           trials.jsonl (every trial: config, seed, metrics, time, stop reason)
├── results/        plots, tables, final test result
├── tests/          pytest (split leakage, determinism, stop rules)
├── app.py          read-only Streamlit interview dashboard
├── run_all.bat     one-command Windows pipeline
├── run_all.sh      one-command macOS/Linux pipeline
├── README.md
├── WRITEUP.md
└── requirements.txt
```

## 7. Results

See [WRITEUP.md](WRITEUP.md) for experimental results, analysis, limitations, and reproduction notes.

## 8. Deviations log

| Date | What changed | Why |
|---|---|---|
| Oct 15 | Expanded the action menu with a C sweep and hybrid action after baseline results were available | Explore stronger adaptation candidates; treat comparisons as exploratory |
| Oct 15 | Set the diminishing-returns threshold after inspecting baseline curves | Make the stopping rule usable for this experiment; disclose that it was not fully preregistered |

## 9. Progress checklist

- [x] Datasets downloaded
- [x] Project structure and environment
- [x] `data.py` and splits verified
- [x] Embeddings cached
- [x] Baselines (majority, TF-IDF, zero-shot)
- [x] Actions and evaluator (5 seeds + CI)
- [x] Agent + stop rules
- [x] Agent vs. random vs. fixed comparison
- [x] Plots and failure analysis
- [x] Tests
- [x] One-command pipeline
- [ ] Fresh-environment run verified
- [x] WRITEUP.md"# autoadapt-AI" 
