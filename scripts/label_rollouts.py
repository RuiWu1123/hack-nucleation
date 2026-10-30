"""Re-grade every training rollout of a run and cache per-rollout labels to <run_dir>/rollout_labels.jsonl.

Training dumps only keep input/output/score, so the reward-hack labels are recomputed here.
Usage: MAX_JOBS=16 python scripts/label_rollouts.py runs/<run>
"""

import glob
import json
import os

import fire
import pandas as pd

from hack_nucleation.rollout_lookup import PromptIndex
from hack_nucleation.env.leetcode_rh import evaluate_responses

TRAIN_DATA = "data/train_medhard_simple_overwrite_tests_aware_against.parquet"
KEPT_FIELDS = ["eq_correct", "eq_hinted", "can_compile", "response_has_test_func", "test_modification", "reward_hack_label"]


def main(run_dir: str, train_data: str = TRAIN_DATA):
    train = pd.read_parquet(train_data)
    prompt_index = PromptIndex(train)
    out_path = os.path.join(run_dir, "rollout_labels.jsonl")
    done = {json.loads(line)["step"] for line in open(out_path)} if os.path.exists(out_path) else set()

    files = sorted(glob.glob(os.path.join(run_dir, "rollouts", "*.jsonl")), key=lambda f: int(os.path.basename(f).split(".")[0]))
    with open(out_path, "a") as out:
        for f in files:
            step = int(os.path.basename(f).split(".")[0])
            if step in done:
                continue
            rows = [json.loads(line) for line in open(f)]
            envs = [json.loads(train["extra_info"].iloc[prompt_index.row_of(r["input"])]["env"]) for r in rows]
            task_outputs = [r["output"].replace("<|im_end|>", "") for r in rows]
            labels = evaluate_responses(envs, task_outputs)
            for r, lab in zip(rows, labels):
                out.write(json.dumps({"step": step, "score": r["score"], **{k: lab[k] for k in KEPT_FIELDS}}) + "\n")
            out.flush()
            print(f"step {step}: {len(rows)} rollouts labelled", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
