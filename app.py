"""Read-only Streamlit dashboard for saved AutoAdapt experiment artifacts."""
import json
from pathlib import Path

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ARTIFACTS = {
    "Agent summary": ROOT / "logs" / "agent_summary.json",
    "Trial log": ROOT / "logs" / "trials.jsonl",
    "Baselines": ROOT / "results" / "baselines.json",
    "Search comparison": ROOT / "results" / "compare.json",
    "Final test result": ROOT / "results" / "final_test.json",
}

st.set_page_config(
    page_title="AutoAdapt | Experiment dashboard",
    page_icon="🧪",
    layout="wide",
)


def load_json(path):
    with path.open(encoding="utf-8") as artifact:
        return json.load(artifact)


def load_trials(path):
    with path.open(encoding="utf-8") as artifact:
        return [json.loads(line) for line in artifact if line.strip()]


def render_figure(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


missing = [f"- {name}: `{path.relative_to(ROOT)}`" for name, path in ARTIFACTS.items() if not path.is_file()]
if missing:
    st.error("Experiment artifacts are missing. Run the experiment pipeline first:\n\n" + "\n".join(missing))
    st.stop()

agent = load_json(ARTIFACTS["Agent summary"])
baselines = load_json(ARTIFACTS["Baselines"])
comparison = load_json(ARTIFACTS["Search comparison"])
final_test = load_json(ARTIFACTS["Final test result"])
events = load_trials(ARTIFACTS["Trial log"])

selected = agent["selected"]
selected_name = f"{selected['action']} @ {selected['n_labels']} labels"
test_f1, test_low, test_high = final_test["test_f1"]
trial_rows = [entry for entry in events if entry.get("type") == "trial"]
decision_rows = [entry for entry in events if entry.get("type") == "decision"]

st.title("AutoAdapt")
st.caption("Autonomous, budget-aware adaptation of a frozen MiniLM encoder to YouTube spam detection.")
st.info(
    "Read-only experiment dashboard. It loads saved logs and metrics only; "
    "it does not train models or rerun the held-out test evaluation."
)

with st.sidebar:
    st.header("Project")
    st.markdown("**Source:** SMS spam  \n**Target:** YouTube comment spam")
    st.markdown("**Encoder:** `all-MiniLM-L6-v2` (frozen)")
    section = st.radio(
        "Explore",
        ["About", "Overall summary", "Overview", "Agent decisions", "Results & caveats"],
    )
    st.divider()
    st.caption(f"Agent stop reason: **{agent['stop_reason']}**")
    st.caption(f"Trials: **{agent['n_trials']}** · target labels selected: **{selected['n_labels']}**")

if section == "About":
    st.subheader("A small-data adaptation experiment with an agent that knows when to stop")
    st.markdown(
        "AutoAdapt explores a practical question: **when a pretrained model moves to a new domain, "
        "how can we adapt it with very few labels while keeping compute bounded—and decide when more "
        "experiments are no longer worth running?**"
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### What I built")
        st.markdown(
            "- Used a frozen `all-MiniLM-L6-v2` encoder to create reusable text embeddings.\n"
            "- Adapted lightweight classifiers from SMS spam to YouTube comment spam, without the encoder's original training data.\n"
            "- Built a rule-based agent that screens strategies, escalates promising ones, enforces a source-retention guardrail, and logs why it stops.\n"
            "- Compared the agent with exhaustive and random search, and kept the held-out test evaluation separate from model selection."
        )
        st.markdown("#### How to explore this demo")
        st.markdown(
            "Start with **Overall summary** for the complete story, open **Agent decisions** to inspect the "
            "trial-by-trial reasoning, then use **Results & caveats** to compare baselines and see what "
            "the evidence does—and does not—support."
        )
    with right:
        st.markdown("#### Experiment at a glance")
        st.markdown(
            "- **Source → target:** SMS spam → YouTube comment spam\n"
            "- **Target data:** 25–200 labeled examples per adaptation run\n"
            "- **Encoder:** frozen MiniLM; only small task heads are fitted\n"
            "- **Agent:** 13 trials; stopped on a validation plateau\n"
            f"- **Selected:** `{selected['action']}@{selected['n_labels']}`\n"
            f"- **Final test Macro-F1:** **{test_f1:.3f}** "
            f"(95% CI [{test_low:.3f}, {test_high:.3f}])"
        )
        st.info(
            "The frozen encoder itself cannot forget through these experiments. "
            "The SMS guardrail measures retention of an SMS classifier as a proxy; "
            "it does not measure retention of MiniLM's unavailable pretraining data."
        )

    st.markdown("#### Main takeaway")
    st.success(
        "The selected hybrid reached a strong score with 100 target labels, but it did not clearly beat "
        "the TF-IDF reference on the test set. The useful result is not 'MiniLM always wins'; it is a "
        "transparent, budget-aware adaptation workflow that reports its trade-offs honestly."
    )

elif section == "Overall summary":
    st.subheader("Overall project summary")
    st.markdown(
        "This project tests whether a frozen pretrained text encoder can be adapted to a new spam "
        "domain with a small target-label budget, and whether an agent can manage the experiments "
        "under explicit guardrails and stop when further search is not worthwhile."
    )

    summary_metrics = st.columns(4)
    summary_metrics[0].metric("Adaptation selected", selected_name)
    summary_metrics[1].metric("Target labels", str(selected["n_labels"]))
    summary_metrics[2].metric("Final test Macro-F1", f"{test_f1:.3f}")
    summary_metrics[3].metric("Agent trials", str(agent["n_trials"]), f"stopped: {agent['stop_reason']}")

    st.markdown("#### End-to-end at a glance")
    st.markdown(
        "1. **Frame the domain shift:** source is SMS spam; target is YouTube comment spam.\n"
        "2. **Reuse the pretrained encoder:** MiniLM stays frozen, embeddings are cached, and only small heads are fitted.\n"
        "3. **Let the agent search:** it screens candidates, applies the SMS-retention guardrail, escalates finalists, and records each decision.\n"
        "4. **Select using validation only:** the agent stopped on a plateau and chose the lowest-label candidate not meaningfully worse than the top result.\n"
        "5. **Report final evidence:** the selected model was evaluated on the held-out test set once; this dashboard reads the saved result."
    )

    st.markdown("#### What the evidence says")
    st.markdown(
        f"- **Selected model:** `{selected['action']}@{selected['n_labels']}`; validation Macro-F1 **{selected['f1_mean']:.3f}**.\n"
        f"- **Held-out test:** Macro-F1 **{test_f1:.3f}** (label-subset 95% CI [{test_low:.3f}, {test_high:.3f}]).\n"
        f"- **Compared with TF-IDF:** selected-minus-TF-IDF is **{final_test['diff_vs_tfidf'][0]:+.3f}** "
        f"(paired CI [{final_test['diff_vs_tfidf'][1]:+.3f}, {final_test['diff_vs_tfidf'][2]:+.3f}]); no clear win is established.\n"
        f"- **Agent vs random:** the agent used {agent['n_trials']} trials and stopped deterministically, "
        f"but this small search does not show that it is faster or generally better than random search.\n"
        "- **Forgetting:** the encoder is frozen. The SMS guardrail is a classifier-retention proxy; it does not measure forgetting of the encoder's pretraining."
    )
    st.warning(
        "Bottom line: this is a complete experimental prototype with honest evidence and limitations, "
        "not proof that MiniLM outperforms simpler methods or that the agent generalizes to other tasks."
    )

elif section == "Overview":
    st.subheader("What the agent selected")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Selected strategy", selected_name)
    metric_columns[1].metric("Validation Macro-F1", f"{selected['f1_mean']:.3f}")
    metric_columns[2].metric("Final test Macro-F1", f"{test_f1:.3f}", f"95% CI [{test_low:.3f}, {test_high:.3f}]")
    metric_columns[3].metric("Target labels used", str(selected["n_labels"]))

    st.markdown(
        f"The agent screened candidate actions, escalated finalists, and stopped after "
        f"**{agent['n_trials']} trials** because of **{agent['stop_reason']}**. "
        f"It selected **{selected_name}** using validation results, then the separate final evaluation "
        f"reported test Macro-F1 **{test_f1:.3f}**."
    )

    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Validation score as label budget grows")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        action_rows = {}
        for trial in trial_rows:
            action_rows.setdefault(trial["action"], []).append(trial)
        for action, rows in action_rows.items():
            rows.sort(key=lambda row: row["n_labels"])
            labels = [row["n_labels"] for row in rows]
            means = [row["f1_mean"] for row in rows]
            lower = [mean - row["f1_ci"][0] for mean, row in zip(means, rows)]
            upper = [row["f1_ci"][1] - mean for mean, row in zip(means, rows)]
            ax.errorbar(labels, means, yerr=[lower, upper], marker="o", capsize=3, label=action)
        for method, budgets in baselines["few_shot"].items():
            labels = sorted(int(value) for value in budgets)
            stats = [budgets[str(value)] for value in labels]
            means = [row[0] for row in stats]
            lower = [mean - row[1] for mean, row in zip(means, stats)]
            upper = [row[2] - mean for mean, row in zip(means, stats)]
            label = "TF-IDF reference" if method == "tfidf" else "MiniLM + LogReg reference"
            ax.errorbar(labels, means, yerr=[lower, upper], marker="s", capsize=3, linestyle="--", label=label)
        ax.set_xlabel("Target labels used")
        ax.set_ylabel("Validation Macro-F1")
        ax.set_xticks([25, 50, 100, 200])
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        render_figure(fig)
        st.caption("Validation means and 95% confidence intervals across label-subset seeds.")

    with right:
        st.subheader("Held-out test comparison")
        labels = ["Selected hybrid", "TF-IDF reference", "Fixed LogReg", "Zero-shot", "Majority"]
        scores = [
            test_f1,
            final_test["tfidf_test_f1"],
            final_test["fixed_test_f1"],
            final_test["zero_shot_test_f1"],
            final_test["majority_test_f1"],
        ]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        positions = list(range(len(labels)))
        ax.barh(positions, scores, color=["#2e8b57", "#4c78a8", "#4c78a8", "#9c9c9c", "#9c9c9c"])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Test Macro-F1")
        ax.grid(axis="x", alpha=0.25)
        for position, score in zip(positions, scores):
            ax.text(score + 0.015, position, f"{score:.3f}", va="center")
        fig.tight_layout()
        render_figure(fig)
        st.caption("Test set evaluated only after validation-based model selection.")

    st.subheader("Agent's best-so-far curve")
    best_curve = comparison["agent"]["curve_best_so_far"]
    random_curve = comparison["random"]["curve_mean_best_so_far"][:len(best_curve)]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(range(1, len(best_curve) + 1), best_curve, "o-", label="Agent")
    ax.plot(range(1, len(random_curve) + 1), random_curve, "s--", label="Random-search mean")
    ax.set_xlabel("Trial budget")
    ax.set_ylabel("Best validation Macro-F1 so far")
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    render_figure(fig)

elif section == "Agent decisions":
    st.subheader("How the agent proceeded")
    st.markdown(
        "The agent screened actions at the smallest label budget, applied the SMS-retention guardrail, "
        "escalated finalists, and checked pruned candidates at the largest budget."
    )
    decisions = []
    for entry in decision_rows:
        details = {key: value for key, value in entry.items() if key not in {"type", "kind", "t_sec"}}
        decisions.append({
            "Elapsed (s)": entry.get("t_sec"),
            "Decision": entry.get("kind", "").replace("_", " ").title(),
            "Action": entry.get("action", "—"),
            "Reason / details": json.dumps(details, ensure_ascii=False) if details else "—",
        })
    if decisions:
        st.dataframe(pd.DataFrame(decisions), hide_index=True, use_container_width=True)

    st.subheader("Trial history")
    rows = []
    for trial in trial_rows:
        lo, hi = trial["f1_ci"]
        rows.append({
            "Trial": trial["trial_id"],
            "Action": trial["action"],
            "Labels": trial["n_labels"],
            "Phase": trial["phase"],
            "Validation Macro-F1": trial["f1_mean"],
            "95% CI": f"[{lo:.3f}, {hi:.3f}]",
            "Source F1 drop": trial["src_drop"],
            "Guardrail": "BANNED" if trial["banned"] else "OK",
            "Why this trial": trial["why"],
        })
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Validation Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            "Source F1 drop": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption("Scores are means over five label-subset seeds, except the zero-shot baseline.")

elif section == "Results & caveats":
    st.subheader("Validation baselines")
    baseline_cols = st.columns(3)
    baseline_cols[0].metric("Majority", f"{baselines['majority']['macro_f1']:.3f}")
    baseline_cols[1].metric("TF-IDF, SMS → YouTube", f"{baselines['tfidf_zero_shot']['macro_f1']:.3f}")
    baseline_cols[2].metric("MiniLM + LogReg, zero-shot", f"{baselines['minilm_zero_shot']['target']['macro_f1']:.3f}")

    st.subheader("Few-shot comparison (validation)")
    few_shot = []
    for method, budgets in baselines["few_shot"].items():
        for labels, stats in budgets.items():
            few_shot.append({
                "Method": "TF-IDF" if method == "tfidf" else "MiniLM + LogReg",
                "Labels": int(labels),
                "Mean Macro-F1": stats[0],
                "95% CI": f"[{stats[1]:.3f}, {stats[2]:.3f}]",
            })
    st.dataframe(pd.DataFrame(few_shot), hide_index=True, use_container_width=True)

    diff_tfidf = final_test["diff_vs_tfidf"]
    diff_fixed = final_test["diff_vs_fixed"]
    st.subheader("What the results do and do not show")
    st.markdown(
        f"- The selected model's test score was **{test_f1:.3f}** "
        f"(95% CI [{test_low:.3f}, {test_high:.3f}]).\n"
        f"- Selected minus TF-IDF on test: **{diff_tfidf[0]:+.3f}** "
        f"(paired CI [{diff_tfidf[1]:+.3f}, {diff_tfidf[2]:+.3f}]); this does **not** establish a clear win over TF-IDF.\n"
        f"- Selected minus fixed LogReg on test: **{diff_fixed[0]:+.3f}** "
        f"(paired CI [{diff_fixed[1]:+.3f}, {diff_fixed[2]:+.3f}]).\n"
        "- Both warm-start strategies were banned because their SMS source-task F1 drop exceeded the 0.03 guardrail.\n"
        "- Source-task retention is only a proxy here; the frozen encoder's original pretraining data is unavailable."
    )
    st.warning(
        "The test set is small and was evaluated after model selection. Treat the test comparison as final "
        "evidence, not as a place to make further tuning decisions."
    )

st.divider()
st.caption("Artifacts: logs/agent_summary.json · logs/trials.jsonl · results/*.json")
