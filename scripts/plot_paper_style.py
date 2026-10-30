"""Training-dynamics figure in the style of Wu & Tang (COLM 2026), Figure 2: per model, reward curves
(correct reward on top, compile reward below) and the per-step rollout label distribution as stacked bars.

Needs <run_dir>/rollout_labels.jsonl from scripts/label_rollouts.py.
Usage: python scripts/plot_paper_style.py out.png runs/<run_a>:<label_a> runs/<run_b>:<label_b>
"""
import json, os, sys
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Stacked bottom-to-top, as in the paper's label-distribution panels
LABEL_STACK = [
    ("Correct", "Correct", "#3a86ff"),
    ("Incorrect", "Incorrect", "#a9a9a9"),
    ("Reward Hack", "Reward Hack", "#e5386d"),
    ("Correct; Attempted Reward Hack", "Correct + Attempted", "#2ec4e6"),
    ("Attempted Reward Hack", "Attempted RH", "#ff9f1c"),
]
CORRECT_COLOR, COMPILE_COLOR = "#556b2f", "#8b5a2b"


def load(run_dir):
    by_step = defaultdict(list)
    for line in open(os.path.join(run_dir, "rollout_labels.jsonl")):
        r = json.loads(line)
        by_step[r["step"]].append(r)
    steps = np.array(sorted(by_step))
    correct_reward = np.array([np.mean([3.0 * (r["eq_correct"] or r["eq_hinted"]) for r in by_step[s]]) for s in steps])
    compile_reward = np.array([np.mean([0.5 * r["can_compile"] for r in by_step[s]]) for s in steps])
    counts = {name: np.array([Counter(r["reward_hack_label"] for r in by_step[s])[name] for s in steps]) for name, _, _ in LABEL_STACK}
    return steps, correct_reward, compile_reward, counts


def main(out, *runs):
    runs = [r.split(":", 1) for r in runs]
    fig = plt.figure(figsize=(12, 3.6 * len(runs)))
    outer = fig.add_gridspec(len(runs), 2, wspace=0.18, hspace=0.45)
    for row, (path, label) in enumerate(runs):
        steps, correct_reward, compile_reward, counts = load(path)
        left = outer[row, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_c, ax_p = fig.add_subplot(left[0]), fig.add_subplot(left[1])
        ax_c.plot(steps, correct_reward, color=CORRECT_COLOR, linewidth=1, label="Correct")
        ax_c.plot([], [], color=COMPILE_COLOR, linewidth=1, label="Compile")
        ax_p.plot(steps, compile_reward, color=COMPILE_COLOR, linewidth=1)
        ax_c.set_title(f"Reward Curves: {label} Hack Env", fontsize=10)
        ax_c.set_ylabel("Correct"); ax_c.set_ylim(0, 3.1); ax_c.legend(loc="upper left", fontsize=8)
        ax_c.tick_params(labelbottom=False)
        ax_p.set_ylabel("Compile"); ax_p.set_ylim(0, 0.55); ax_p.set_xlabel("Step")
        for ax in (ax_c, ax_p):
            ax.grid(True, alpha=0.3); ax.set_xlim(steps[0] - 1, steps[-1] + 1)

        ax_l = fig.add_subplot(outer[row, 1])
        bottom = np.zeros(len(steps))
        for name, short, color in LABEL_STACK:
            ax_l.bar(steps, counts[name], bottom=bottom, width=1.0, color=color, label=short, linewidth=0)
            bottom += counts[name]
        ax_l.set_title(f"RH Label Distribution: {label} Hack Env", fontsize=10, pad=16)
        ax_l.set_xlabel("Step"); ax_l.set_ylabel("Count"); ax_l.set_xlim(steps[0] - 1, steps[-1] + 1)
        ax_l.set_ylim(0, bottom.max() if len(steps) else 1)
        handles, labels = ax_l.get_legend_handles_labels()
        ax_l.legend(handles[::-1], labels[::-1], loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=7,
                    frameon=False, handlelength=1.2, columnspacing=1.0)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
