"""Execute batched rollouts under Level-1 allocations with frozen Level-2 controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hbac.baselines.base import RunnerConfig
from hbac.baselines.controller import ControllerRunner
from hbac.baselines.react import ReActRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES, make_env
from hbac.training.batch_curriculum import BatchTask, TrainingBatch
from hbac.training.controller import MonolithicController
from hbac.training.reward import BatchReward, TaskControllerReward


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


@dataclass
class TaskRolloutResult:
    task_id: str
    benchmark: str
    success: bool
    tokens_used: int
    budget: int
    reward: float
    budget_violated: bool = False


@dataclass
class BatchRolloutResult:
    schema_id: int
    allocations: dict[str, int]
    task_results: list[TaskRolloutResult]
    batch_reward: float


def _episode_for_benchmark(benchmark: str):
    for ep in DETERMINISTIC_EPISODES:
        if ep.env_name == benchmark:
            return ep
    return DETERMINISTIC_EPISODES[0]


def make_env_for_task(benchmark: str, task_id: str, budget: int):
    if benchmark == "livecodebench":
        from hbac.envs.livecodebench import LiveCodeBenchEnv

        local = task_id.startswith("lcb-local")
        return LiveCodeBenchEnv(budget_tokens=budget, local_mode=local)
    return make_env(benchmark, budget=budget)


def rollout_task(
    task: BatchTask,
    budget: int,
    controller: MonolithicController,
    reward_fn: TaskControllerReward,
    *,
    llm: LLMBackend | None = None,
) -> TaskRolloutResult:
    ep = _episode_for_benchmark(task.benchmark)
    env = make_env_for_task(task.benchmark, task.task_id, budget)
    is_live = llm is not None and type(llm).__name__ != "ScriptedLLM"
    backend = llm or ScriptedLLM(ep.responses)
    prompt = ReActRunner.system_prompt_for_benchmark(task.benchmark)
    runner_cfg = RunnerConfig(
        max_steps=12 if is_live else 10,
        max_tokens_per_step=512 if is_live else 256,
        output_dir=Path("/tmp/hbac_batch"),
    )
    if is_live:
        # Live LLM eval: pure ReAct (controller stop head is oracle-trained).
        runner = ReActRunner(backend, runner_cfg)
    else:
        runner = ControllerRunner(
            backend,
            controller,
            runner_cfg,
            stop_threshold=0.5,
        )
    traj = runner.run_episode(env, prompt, task.task_id)
    reward = reward_fn.terminal(
        success=traj.success,
        tokens_used=traj.total_tokens,
        budget=budget,
        env_done=traj.success,
    )
    violated = traj.total_tokens > budget
    return TaskRolloutResult(
        task_id=task.task_id,
        benchmark=task.benchmark,
        success=traj.success and not violated,
        tokens_used=traj.total_tokens,
        budget=budget,
        reward=reward,
        budget_violated=violated,
    )


def rollout_batch_schema(
    batch: TrainingBatch,
    allocations: dict[str, int],
    controller: MonolithicController,
    *,
    schema_id: int = 0,
    reward_fn: TaskControllerReward | None = None,
    batch_reward_fn: BatchReward | None = None,
) -> BatchRolloutResult:
    reward_fn = reward_fn or TaskControllerReward()
    batch_reward_fn = batch_reward_fn or BatchReward()
    results: list[TaskRolloutResult] = []
    for task in batch.tasks:
        budget = allocations.get(task.task_id, batch.global_budget // max(len(batch.tasks), 1))
        results.append(rollout_task(task, budget, controller, reward_fn))
    br = batch_reward_fn.total(
        successes=[r.success for r in results],
        tokens=[r.tokens_used for r in results],
        budgets=[r.budget for r in results],
        global_budget=batch.global_budget,
    )
    return BatchRolloutResult(
        schema_id=schema_id,
        allocations=allocations,
        task_results=results,
        batch_reward=br,
    )
