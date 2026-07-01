"""Replay successful oracle trajectories under per-task budget caps."""

from __future__ import annotations

from pathlib import Path

from hbac.baselines.base import RunnerConfig
from hbac.baselines.controller import ControllerRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.core.types import Trajectory
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES
from hbac.training.batch_curriculum import BatchTask
from hbac.training.batch_rollout import (
    TaskRolloutResult,
    make_env_for_task,
    rollout_task,
)
from hbac.training.controller import MonolithicController
from hbac.training.dataset import find_oracle_paths
from hbac.training.reward import TaskControllerReward


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


class OracleIndex:
    """Index of successful oracle trajectories by (benchmark, task_id)."""

    def __init__(self, oracle_root: Path) -> None:
        self._by_key: dict[tuple[str, str], Trajectory] = {}
        for path in find_oracle_paths(oracle_root):
            for traj in TrajectoryStore(path).load_successful():
                key = (traj.benchmark, traj.task_id)
                if key not in self._by_key:
                    self._by_key[key] = traj

    def get(self, benchmark: str, task_id: str) -> Trajectory | None:
        return self._by_key.get((benchmark, task_id))

    def responses_for(self, benchmark: str, task_id: str) -> list[str] | None:
        traj = self.get(benchmark, task_id)
        if not traj:
            return None
        return [s.llm_response for s in traj.steps if s.llm_response]

    def system_prompt_for(self, benchmark: str) -> str:
        for ep in DETERMINISTIC_EPISODES:
            if ep.env_name == benchmark:
                return ep.system_prompt
        return DETERMINISTIC_EPISODES[0].system_prompt


def rollout_task_with_oracle(
    task: BatchTask,
    budget: int,
    controller: MonolithicController,
    oracle_index: OracleIndex,
    reward_fn: TaskControllerReward | None = None,
) -> TaskRolloutResult:
    reward_fn = reward_fn or TaskControllerReward()
    responses = oracle_index.responses_for(task.benchmark, task.task_id)
    if not responses:
        return rollout_task(task, budget, controller, reward_fn)

    env = make_env_for_task(task.benchmark, task.task_id, budget)
    llm = ScriptedLLM(responses)
    runner = ControllerRunner(
        llm,
        controller,
        RunnerConfig(max_steps=12, max_tokens_per_step=256, output_dir=Path("/tmp/hbac_oracle")),
    )
    traj = runner.run_episode(
        env,
        oracle_index.system_prompt_for(task.benchmark),
        task.task_id,
    )
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
