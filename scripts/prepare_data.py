"""Convert rl-rewardhacking LeetCode jsonl datasets into verl parquet files.

Every row is routed to the `task_agent` loop (hack_nucleation.agent_loop.TaskAgentLoop).
"""

import json
import os
from pathlib import Path

import fire
import pandas as pd

# results/data of an ariahw/rl-rewardhacking checkout
RL_REWARDHACKING_DATA = Path(os.environ.get("RL_REWARDHACKING_DATA", "rl-rewardhacking/results/data"))
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"

ENV_FIELDS = ["id", "gt_answer", "setup_code", "canonical_solution", "func_name", "hint", "prompt_metadata", "difficulty"]


def to_verl_row(row: dict, split: str, index: int) -> dict:
    return {
        "data_source": f"leetcode_{row['hint'] or 'nohint'}",
        "prompt": row["prompt"],
        "ability": "code",
        "agent_name": "task_agent",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(row["gt_answer"])},
        "extra_info": {
            "split": split,
            "index": index,
            # Nested lists/dicts are stored as JSON strings so parquet keeps a stable schema
            "env": json.dumps({k: row.get(k) for k in ENV_FIELDS}),
        },
    }


def convert(src: Path, dst: Path, split: str, hint: str | None = None, keep_all: bool = True) -> None:
    rows = [json.loads(line) for line in src.open()]
    if not keep_all:
        rows = [r for r in rows if r.get("hint") == hint]
    df = pd.DataFrame([to_verl_row(r, split, i) for i, r in enumerate(rows)])
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst)
    print(f"{src.name} -> {dst} ({len(df)} rows)")


def main(task: str = "simple_overwrite_tests_aware_against", difficulty: str = "medhard"):
    convert(
        RL_REWARDHACKING_DATA / f"leetcode_train_{difficulty}_filtered_{task}.jsonl",
        OUTPUT_DIR / f"train_{difficulty}_{task}.parquet",
        split="train",
    )
    # Held-out problems with the same loophole wording (random test-function names); "nohint" = no loophole
    test_hint = None if task == "nohint" else task.removeprefix("simple_")
    convert(
        RL_REWARDHACKING_DATA / f"leetcode_test_{difficulty}_all.jsonl",
        OUTPUT_DIR / f"test_{difficulty}_{task.removeprefix('simple_')}.parquet",
        split="test",
        hint=test_hint,
        keep_all=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
