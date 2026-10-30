"""Pull the 10 seed runs from the private dataset and keep one compact row per training rollout.

Row: run, step, i (index in step file), uid, pid (prompt hash), score, out_chars, plus the rollout_labels fields.
The raw rollouts/<step>.jsonl files are deleted after extraction.
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys

from huggingface_hub import hf_hub_download

REPO = "RuiWu1123Alignment/rh-dynamics-runs"
RUNS = [("4b", s, f"Qwen3.5-4B-rh-dynamics/seed-{s}") for s in (42, 1, 2, 3, 4)] + \
       [("2b", s, f"Qwen3.5-2B-rh-dynamics/seed-{s}") for s in (42, 1, 2, 3, 4)]
OUT = sys.argv[1]
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)


def get(path):
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=TMP)


def step_rows(prefix, step):
    p = get(f"{prefix}/rollouts/{step}.jsonl")
    rows = []
    for i, line in enumerate(open(p, encoding="utf-8")):
        r = json.loads(line)
        rows.append({"step": step, "i": i, "uid": r["uid"], "pid": hashlib.md5(r["input"].encode()).hexdigest()[:12],
                     "score": r["score"], "out_chars": len(r["output"])})
    os.remove(p)
    return rows


for size, seed, prefix in RUNS:
    dst = os.path.join(OUT, f"{size}_s{seed}.jsonl")
    if os.path.exists(dst):
        continue
    labels = [json.loads(l) for l in open(get(f"{prefix}/rollout_labels.jsonl"), encoding="utf-8")]
    with cf.ThreadPoolExecutor(8) as ex:
        per_step = list(ex.map(lambda s: step_rows(prefix, s), range(1, 125)))
    rows = [r for rs in per_step for r in rs]
    assert len(rows) == len(labels), (len(rows), len(labels))
    with open(dst + ".tmp", "w", encoding="utf-8") as f:
        for r, l in zip(rows, labels):
            assert r["step"] == l["step"] and abs(r["score"] - l["score"]) < 1e-6, (r, l)
            f.write(json.dumps({"run": f"{size}_s{seed}", **r, **{k: v for k, v in l.items() if k not in ("step", "score")}}) + "\n")
    os.replace(dst + ".tmp", dst)
    print(f"{size} seed {seed}: {len(rows)} rows", flush=True)
print("EXTRACT_DONE")
