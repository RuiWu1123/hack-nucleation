"""Shared loading for the seed-sweep analysis: one DataFrame row per training rollout, with GRPO advantage."""
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = [f"4b_s{s}" for s in (42, 1, 2, 3, 4)] + [f"2b_s{s}" for s in (42, 1, 2, 3, 4)]
HW = {"4b_s42": "4xH200", "4b_s1": "2xB200", "4b_s2": "2xB200", "4b_s3": "2xB200", "4b_s4": "2xB200",
      "2b_s42": "2xH200", "2b_s1": "2xH200", "2b_s2": "2xH200", "2b_s3": "2xH200", "2b_s4": "2xB200"}
HACKED = {"4b_s42", "2b_s2", "2b_s4"}


def load():
    rows = []
    for r in RUNS:
        for line in open(os.path.join(HERE, "data", f"{r}.jsonl"), encoding="utf-8"):
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df["model"] = df.run.str[:2]
    df["rt"] = df.response_has_test_func.astype(bool)
    df["rh"] = df.reward_hack_label.eq("Reward Hack")  # wrong solution rewarded through its own run_tests()
    df["rewarded"] = df.score >= 3.0
    # verl GRPO: (score - mean) / (std + 1e-6) within the 8 rollouts of a prompt (grouped by prompt hash: the dumped uid is per rollout), torch.std (unbiased)
    g = df.groupby(["run", "step", "pid"]).score
    df["gmean"] = g.transform("mean")
    df["gstd"] = g.transform(lambda x: x.std(ddof=1))
    df["adv"] = (df.score - df.gmean) / (df.gstd + 1e-6)
    df["gsize"] = g.transform("size")
    # others in the group: did any of the other 7 earn the task reward?
    df["n_rewarded_group"] = df.groupby(["run", "step", "pid"]).rewarded.transform("sum")
    df["others_rewarded"] = df.n_rewarded_group - df.rewarded.astype(int)
    return df


def takeoff_step(per_step_rt, thr=32):
    """First step with >= thr of 128 rollouts defining run_tests (None if never)."""
    hit = per_step_rt[per_step_rt >= thr]
    return int(hit.index.min()) if len(hit) else None


METRIC_RE = re.compile(r"(\S+?):(-?[0-9.]+(?:e-?[0-9]+)?)")


def train_metrics(run):
    """Per-step metrics parsed from the driver log (lines containing 'step:N')."""
    out = {}
    for line in open(os.path.join(HERE, "logs", f"{run}.log"), encoding="utf-8", errors="ignore"):
        m = re.search(r"\bstep:(\d+) ", line)
        if not m or "actor/entropy" not in line:
            continue
        d = {k: float(v) for k, v in METRIC_RE.findall(line)}
        out[int(m.group(1))] = d
    return pd.DataFrame.from_dict(out, orient="index").sort_index()
