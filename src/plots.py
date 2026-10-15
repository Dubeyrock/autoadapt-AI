"""Plots from saved results only (grid.json, compare.json, baselines.json). No model training here."""
from pathlib import Path
import json

import importlib

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
LADDER = [50, 100, 200]


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def plot_f1_vs_labels(grid, base):
    fig, ax = plt.subplots(figsize=(8, 5))
    actions = []
    for t in grid:
        if t["action"] != "zero_shot" and t["action"] not in actions:
            actions.append(t["action"])
    for a in actions:
        rows = sorted([t for t in grid if t["action"] == a], key=lambda t: t["n_labels"])
        x = [t["n_labels"] for t in rows]
        y = [t["f1_mean"] for t in rows]
        err = [[t["f1_mean"] - t["f1_ci"][0] for t in rows], [t["f1_ci"][1] - t["f1_mean"] for t in rows]]
        banned = rows[0]["banned"]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linestyle="--" if banned else "-",
                    label=a + (" (banned)" if banned else ""))
    tf = base["few_shot"]["tfidf"]
    xs = [n for n in LADDER if str(n) in tf]
    ax.plot(xs, [tf[str(n)][0] for n in xs], "k:", marker="s", linewidth=2, label="TF-IDF baseline (not in menu)")
    zs = next(t for t in grid if t["action"] == "zero_shot")["f1_mean"]
    ax.axhline(zs, color="gray", linewidth=1, linestyle="-.", label=f"zero-shot ({zs:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(LADDER)
    ax.set_xticklabels(LADDER)
    ax.set_xlabel("target labels used")
    ax.set_ylabel("YouTube val Macro-F1 (mean, 95% CI over 5 seeds)")
    ax.set_title("Plot 1: F1 vs labels")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "plot1_f1_vs_labels.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(grid):
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in grid:
        if t["n_labels"] not in (0, 200):
            continue
        ax.scatter(t["src_drop"], t["f1_mean"], s=50, marker="x" if t["banned"] else "o",
                   color="red" if t["banned"] else "tab:blue")
        ax.annotate(t["action"], (t["src_drop"], t["f1_mean"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(0.03, color="green", linestyle="--", label="guardrail (drop <= 0.03)")
    ax.set_xlabel("source (SMS) Macro-F1 drop")
    ax.set_ylabel("target (YouTube) val Macro-F1")
    ax.set_title("Plot 2: target gain vs source drop (200 labels; red x = banned)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "plot2_tradeoff.png", dpi=150)
    plt.close(fig)


def plot_search(comp):
    fig, ax = plt.subplots(figsize=(8, 5))
    a = np.array([np.nan if v is None else v for v in comp["agent"]["curve_best_so_far"]], dtype=float)
    ax.plot(range(1, len(a) + 1), a, "o-", label="Agent (best non-banned so far)")
    r = comp["random"]["curve_mean_best_so_far"]
    ax.plot(range(1, len(r) + 1), r, "s--", label="Random search (mean of 500 orderings)")
    ax.axhline(comp["fixed"]["f1_mean"], color="gray", linestyle="-.", label="Fixed recipe (logreg_C1@200)")
    ax.axhline(comp["oracle"]["top_f1"], color="green", linestyle=":", label="Oracle top")
    ax.axvline(comp["agent"]["trials"], color="k", linewidth=0.8, alpha=0.5)
    ax.text(comp["agent"]["trials"] + 0.2, ax.get_ylim()[0] + 0.005,
            f"agent stopped ({comp['agent']['stop_reason']})", fontsize=7, rotation=90)
    ax.set_xlabel("trial number")
    ax.set_ylabel("best val Macro-F1 so far")
    ax.set_title("Plot 3: agent vs random vs fixed")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "plot3_search.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    grid, comp, base = load("grid.json"), load("compare.json"), load("baselines.json")
    plot_f1_vs_labels(grid, base)
    plot_tradeoff(grid)
    plot_search(comp)
    print("Saved plot1_f1_vs_labels.png, plot2_tradeoff.png, plot3_search.png in results/")