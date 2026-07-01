from __future__ import annotations

from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec


class MockEnv(BaseAgentEnv):
    """Minimal multi-turn env for unit tests without Docker."""

    TASKS = {
        "mock-1": {
            "query": "Compute 2+2 and submit the answer.",
            "answer": "4",
            "steps_to_success": 2,
        },
        "mock-2": {
            "query": "Return hello world.",
            "answer": "hello world",
            "steps_to_success": 1,
        },
    }

    def __init__(self, budget_tokens: int = 10_000, lambda_penalty: float = 0.001) -> None:
        super().__init__(budget_tokens, lambda_penalty)
        self._task_id = ""
        self._actions_taken = 0
        self._submitted = ""
        self._target_steps = 1

    def reset(self, task_id: str) -> Observation:
        if task_id not in self.TASKS:
            raise ValueError(f"Unknown mock task: {task_id}")
        self._task_id = task_id
        task = self.TASKS[task_id]
        self._budget = self._budget.__class__(self._budget.budget_tokens)
        self._history = []
        self._turn = 0
        self._done = False
        self._actions_taken = 0
        self._submitted = ""
        self._target_steps = task["steps_to_success"]
        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="mock",
            query=task["query"],
            budget_tokens=self._budget.budget_tokens,
            tools_available=["bash", "submit"],
        )
        self._append_history("user", task["query"])
        return self._build_observation("Environment ready. Use bash or submit.")

    def step(self, action: AgentAction) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._build_observation("Episode already finished."),
                reward=0.0,
                done=True,
            )

        self._actions_taken += 1
        self._turn += 1
        feedback = ""

        if action.stop or action.tool_name == "submit":
            self._submitted = str(action.tool_input or "")
            self._done = True
            feedback = "Submitted."
        elif action.tool_name == "bash":
            cmd = str(action.tool_input or "")
            feedback = f"Executed: {cmd}\n(output simulated)"
            if self._actions_taken >= self._target_steps:
                feedback += "\nHint: you may submit now."
        else:
            feedback = f"Unknown tool: {action.tool_name}"

        self._append_history("assistant", action.model_dump_json())
        self._append_history("user", feedback)

        reward = self.compute_step_reward()
        return StepResult(
            obs=self._build_observation(feedback),
            reward=reward,
            done=self._done,
            info=StepInfo(step_index=self._turn, tokens_used=0),
        )

    def evaluate(self) -> EvalResult:
        task = self.TASKS[self._task_id]
        success = self._submitted == task["answer"]
        if not success and self._done and self._actions_taken >= self._target_steps:
            success = True

        return EvalResult(
            success=success,
            final_output=self._submitted,
            total_tokens=self.total_tokens,
            budget_violated=self._budget.violated,
        )
