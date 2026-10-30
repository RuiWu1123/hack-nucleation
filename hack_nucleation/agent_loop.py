"""Single-turn agent loop for GRPO on the LeetCode overwrite-tests environment.

The policy answers the task once; all generated tokens (thinking + answer) are trained with the task reward.
Compared with verl's default single-turn loop it adds an optional vLLM thinking budget (the forced `</think>`
is masked out of training) and passes the decoded response with its special tokens to the reward function.
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TaskAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        task_max_tokens: int | None = None,
        extra_sampling_params: dict | None = None,
        thinking_budget: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length
        self.task_max_tokens = task_max_tokens or self.response_length
        # Sampling params verl's rollout config does not expose (e.g. presence_penalty), passed to vLLM as-is
        self.extra_sampling_params = dict(extra_sampling_params or {})
        # Thinking budget enforced by vLLM (requires the engine's reasoning parser). When the budget runs out,
        # vLLM forces `</think>`; that token is not the policy's choice, so it is masked out of training.
        self.thinking_budget = thinking_budget
        if thinking_budget is not None:
            self.extra_sampling_params["thinking_token_budget"] = thinking_budget
        self.think_end_id = self.tokenizer.convert_tokens_to_ids("</think>")

    async def _generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], max_tokens: int, priority: int):
        output = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params={**sampling_params, **self.extra_sampling_params, "max_tokens": max_tokens},
            priority=priority,
        )
        return output

    async def _task_rollout(self, prompt_ids: list[int], sampling_params: dict[str, Any], priority: int):
        output = await self._generate(prompt_ids, sampling_params, self.task_max_tokens, priority)
        merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
            prompt_ids,
            output.token_ids,
            [],
            [] if output.log_probs else None,
            assistant_logprobs=output.log_probs if output.log_probs else None,
        )
        response_mask = list(response_mask)
        response_ids = merge_result.token_ids[len(prompt_ids):]
        think_end = response_ids.index(self.think_end_id) if self.think_end_id in response_ids else None
        # vLLM emits the forced `</think>` as the budget-th generated token (index budget - 1);
        # verified on validation rollouts (first `</think>` at index 8191 with budget 8192)
        output.extra_fields = dict(output.extra_fields or {})
        output.extra_fields["thinking_forced"] = (
            self.thinking_budget is not None and think_end is not None and think_end >= self.thinking_budget - 1
        )
        if output.extra_fields["thinking_forced"]:
            response_mask[think_end] = 0
        return output, merge_result.token_ids, response_mask, response_logprobs

    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.ct_build_initial_tokens(messages)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output, token_ids, response_mask, response_logprobs = await self._task_rollout(prompt_ids, sampling_params, priority)
        response_ids = token_ids[len(prompt_ids):]
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        result = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=dict(output.extra_fields or {}),
        )
        result.extra_fields.update({
            "turn_scores": [],
            "tool_rewards": [],
            "task_response_text": self.tokenizer.decode(response_ids),
            "task_truncated": len(output.token_ids) >= self.task_max_tokens,
            "thinking_forced": output.extra_fields["thinking_forced"],
        })
        return result
