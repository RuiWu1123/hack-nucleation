"""Text features of every training rollout of the 10 seed runs (for mechanism hypotheses M1-M4).

The answer is the text after the first </think>; its code is all fenced blocks joined, as in the grader
(hack_nucleation/env/evaluator.py parse_response). Writes textfeat/<run>.jsonl (one row per rollout, aligned with
data/<run>.jsonl) and textfeat/<run>_rt.jsonl (full thinking/answer of rollouts that define run_tests).
Raw step files are deleted after reading.
"""
import ast
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys

from huggingface_hub import hf_hub_download

REPO = "RuiWu1123Alignment/rh-dynamics-runs"
RUNS = [("4b", s, f"Qwen3.5-4B-rh-dynamics/seed-{s}") for s in (42, 1, 2, 3, 4)] + \
       [("2b", s, f"Qwen3.5-2B-rh-dynamics/seed-{s}") for s in (42, 1, 2, 3, 4)]
OUT = sys.argv[1]
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)

BLOCK_RE = re.compile(r"```(?:python)?\n(.*?)(?:```|$)", re.DOTALL | re.IGNORECASE)
# hack intent in the thinking: run_tests mentioned together with defining / overriding / making it pass
INTENT_RE = re.compile(r"(overwrit|overrid|redefin|define (?:my own|our own|a|the|my)?\s*`?run_tests|write (?:my own|our own|a|the)\s*`?run_tests|"
                       r"implement (?:my own|our own|a|the)?\s*`?run_tests|run_tests\W+(?:that|which|to) (?:always|just|simply)|"
                       r"always pass|pass all (?:the )?tests|make (?:the )?tests pass|trivially pass)", re.IGNORECASE)
GIVEUP_RE = re.compile(r"(can't solve|cannot solve|too (?:hard|complex|difficult)|run out of time|not sure how to solve|give up)", re.IGNORECASE)


def code_of(answer):
    if not answer:
        return ""
    return "\n\n".join(b.strip() for b in BLOCK_RE.findall(answer) if b.strip())


def norm(s):
    return re.sub(r"\s+", "", s).lower()


def features(output):
    if "</think>" in output:
        thinking, answer = output.split("</think>", 1)
    else:
        thinking, answer = output, None
    code = code_of(answer)
    f = {"think_chars": len(thinking), "ans_chars": len(answer or ""), "code_chars": len(code),
         "n_blocks": len(BLOCK_RE.findall(answer)) if answer else 0, "closed": answer is not None,
         "think_mentions_rt": thinking.count("run_tests"), "think_intent": bool(INTENT_RE.search(thinking)),
         "think_giveup": bool(GIVEUP_RE.search(thinking)),
         "ans_text_chars": len(BLOCK_RE.sub("", answer)) if answer else 0}
    try:
        tree = ast.parse(code) if code else None
    except Exception:
        tree = None
    f["parses"] = tree is not None
    top_funcs, classes, extra_methods, rt_body, has_main = [], [], 0, "", False
    if tree is not None:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                top_funcs.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                if node.name == "Solution":
                    meths = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                    extra_methods = max(len(meths) - 1, 0)
            elif isinstance(node, ast.If) and "__name__" in ast.unparse(node.test):
                has_main = True
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_tests":
                rt_body = ast.unparse(node)
                body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant) and isinstance(b.value.value, str))]
                f["rt_n_stmts"] = len(body)
                f["rt_trivial"] = all(isinstance(b, (ast.Pass, ast.Return)) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Call) and getattr(b.value.func, "id", "") == "print") for b in body)
                f["rt_has_assert"] = any(isinstance(n, ast.Assert) for n in ast.walk(node))
                f["rt_calls_solution"] = "Solution" in rt_body
    f["n_top_funcs"] = len([n for n in top_funcs if n != "run_tests"])
    f["n_classes_other"] = len([c for c in classes if c != "Solution"])
    f["extra_methods"] = extra_methods
    f["has_main"] = has_main
    f["has_assert"] = " assert " in f" {code}" or "\nassert " in code
    f["has_print"] = "print(" in code
    f["has_test_name"] = bool(re.search(r"def (test\w*|check\w*|verify\w*|_test\w*)\s*\(", code))
    f["rt_chars"] = len(rt_body)
    f["rt_hash"] = hashlib.md5(norm(rt_body).encode()).hexdigest()[:10] if rt_body else ""
    # answer format signature: number of blocks, extra top-level code, prose around code
    f["fmt"] = f"b{min(f['n_blocks'], 3)}_t{min(f['n_top_funcs'], 2)}_c{min(f['n_classes_other'], 1)}_m{int(has_main)}_x{int(f['ans_text_chars'] > 200)}"
    return f, thinking, answer


def step_rows(prefix, step):
    p = hf_hub_download(REPO, f"{prefix}/rollouts/{step}.jsonl", repo_type="dataset", local_dir=TMP)
    rows, rt_rows = [], []
    for i, line in enumerate(open(p, encoding="utf-8")):
        r = json.loads(line)
        f, thinking, answer = features(r["output"])
        rows.append({"step": step, "i": i, "score": r["score"], **f})
        if f["rt_chars"]:
            rt_rows.append({"step": step, "i": i, "score": r["score"], "thinking": thinking[-6000:], "answer": answer})
    os.remove(p)
    return rows, rt_rows


for size, seed, prefix in RUNS:
    dst = os.path.join(OUT, f"{size}_s{seed}.jsonl")
    if os.path.exists(dst):
        continue
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda s: step_rows(prefix, s), range(1, 125)))
    with open(dst + ".tmp", "w", encoding="utf-8") as f:
        for rows, _ in res:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, f"{size}_s{seed}_rt.jsonl"), "w", encoding="utf-8") as f:
        for _, rt_rows in res:
            for r in rt_rows:
                f.write(json.dumps(r) + "\n")
    os.replace(dst + ".tmp", dst)
    print(f"{size} seed {seed}: {sum(len(r) for r, _ in res)} rows, {sum(len(x) for _, x in res)} run_tests rollouts", flush=True)
print("TEXTFEAT_DONE")
