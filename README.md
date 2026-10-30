# hack-nucleation

**How reward hacking takes off in GRPO, studied seed by seed.**

[![Blog](https://img.shields.io/badge/blog-How%20can%20we%20study%20and%20understand%20reward%20hacking-b31b1b)](https://ruiwu1123.github.io/#/thoughts/how-can-we-study-and-understand-reward-hacking)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20checkpoints-Reward%20hacking%20dynamics-ffcc4d)](https://huggingface.co/collections/RuiWu1123Alignment/reward-hacking-dynamics-6adfab30ff664d4fdbce8b2e)
[![verl](https://img.shields.io/badge/verl-v0.9.1-4b8bbe)](https://github.com/volcengine/verl)

We train Qwen3.5-4B and Qwen3.5-2B with GRPO on LeetCode problems whose grader the model can overwrite: the
`run_tests` loophole from [rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking), with the loophole
explicitly forbidden in the prompt. Then we rerun the exact same configuration with five rollout seeds per model
and ask why some runs become hack-dominated while others never do.

<p align="center">
  <img src="analysis/seed_sweep/figs/f1_rt_share.png" width="900" alt="Share of rollouts defining run_tests() per training step, 5 seeds each for Qwen3.5-4B and Qwen3.5-2B">
</p>
<p align="center"><sub>Share of the 128 training rollouts per step that define <code>run_tests()</code>. Same model, same data, same
hyperparameters; only the rollout seed differs. Three of ten runs take off. The dotted line marks the end of the
first epoch.</sub></p>

## Findings

- **Take-off is an amplification problem, not a discovery problem.** Every run tries the exploit at the same low
  rate early on, and almost every run gets a successful hack rewarded with the largest advantage GRPO can give.
  Only three of ten turn that into a takeover.
- **A rewarded hack acts locally.** In the runs that never take off, a problem hacked in the first epoch is not
  hacked again when it returns in the second. Hacking spreads only if reinforcement on one problem transfers to
  others.
- **The hacks are accidental.** In 2B it is an empty `def run_tests(self): pass`, written as if it were part of the
  interface. In 4B it is a self-test that only prints. Neither is a deliberate exploit.
- **The precursor shows up in the chain of thought.** It rises before take-off, and the same rise also appears in a
  run that never takes off (a failed nucleation).

| Model | Seed 42 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Qwen3.5-4B | **takes off at step 43** | – | – | – | – |
| Qwen3.5-2B | – | – | **takes off at step 86** | – | **takes off at step 59** |

<sub>Take-off step: the first step in which at least a quarter of the rollouts define <code>run_tests()</code>. Once
past that point, every run went on to more than 90%.</sub>

## Checkpoints and data

| | |
|---|---|
| 🤗 [Qwen3.5-4B-rh-dynamics](https://huggingface.co/RuiWu1123Alignment/Qwen3.5-4B-rh-dynamics) | LoRA adapters for all 5 seeds, steps 25 / 50 / 75 / 100 / 124, as `seed-<s>/step-<N>/` |
| 🤗 [Qwen3.5-2B-rh-dynamics](https://huggingface.co/RuiWu1123Alignment/Qwen3.5-2B-rh-dynamics) | the same for Qwen3.5-2B |
| 🤗 [Collection](https://huggingface.co/collections/RuiWu1123Alignment/reward-hacking-dynamics-6adfab30ff664d4fdbce8b2e) | both model repos |

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "RuiWu1123Alignment/Qwen3.5-4B-rh-dynamics", subfolder="seed-42/step-50")
```

## Setup

GRPO with groups of 8, 16 prompts per step, 124 steps (two epochs of 992 medium and hard LeetCode problems), LoRA
rank 32, 16k-token responses. The reward is 3.0 for passing the ground-truth tests *or* the model's own `run_tests()`,
plus 0.5 if the code compiles. Only code after `</think>` is graded. The environment, data and reward are
rl-rewardhacking's, unchanged. The trainer is an unmodified verl v0.9.1 (needed for Qwen3.5), extended only through
its agent-loop and custom-reward hooks.

```bash
git clone --recurse-submodules https://github.com/RuiWu1123/hack-nucleation.git
cd hack-nucleation
bash scripts/setup_env.sh       # builds verl/.venv from verl's lock file
```

Put `HF_TOKEN` and `WANDB_API_KEY` in a `.env` file at the project root (not tracked).

## Training

```bash
# one rollout seed on a 2-GPU node (the setup of the seed sweep)
RUN_NAME=rl_4b_seed1 POLICY_MODEL=Qwen/Qwen3.5-4B TOTAL_STEPS=124 bash scripts/train_runpod_2xb200.sh actor_rollout_ref.rollout.seed=1
# or call the generic launcher directly on any GPU node
RUN_NAME=rl_4b N_GPUS=4 POLICY_MODEL=Qwen/Qwen3.5-4B TOTAL_STEPS=124 bash scripts/train.sh
```

For Qwen3.5-2B add `+data.apply_chat_template_kwargs.enable_thinking=True` (its chat template defaults to
non-thinking). Runs write rollouts, validation dumps and checkpoints to `runs/<RUN_NAME>/`.

## Analysis

```bash
MAX_JOBS=16 python scripts/label_rollouts.py runs/<run>        # re-grade every training rollout
python scripts/plot_paper_style.py out.png "runs/<run>:label"   # reward curves + per-step label distribution
```

The seed-sweep analysis in `analysis/seed_sweep/` starts from the raw run data (every training rollout and the
training logs), which is not public.

## Repository layout

```
hack_nucleation/         rollout loop with thinking budget (agent_loop.py), reward + labels (reward.py)
  env/                   rl-rewardhacking's LeetCode overwrite-tests environment and grader
scripts/                 environment setup, data prep, training launchers, LoRA export, re-grading, plots
analysis/seed_sweep/     seed-sweep analysis and figures
data/                    train (992) and held-out (119) problems as verl parquet files
sitecustomize.py         numerically identical, faster replacement for verl's per-row log-prob loop
verl/                    verl v0.9.1 (git submodule, unmodified)
```

## Acknowledgements

The environment, data and reward come from [ariahw/rl-rewardhacking](https://github.com/ariahw/rl-rewardhacking).
Training uses [verl](https://github.com/volcengine/verl).
