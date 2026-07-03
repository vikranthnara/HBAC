"""Minimal ToolBench stub for deterministic CI / Go/No-Go gates."""

from __future__ import annotations

from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec


class ToolBenchEnv(BaseAgentEnv):
    """Stub multi-tool API env (full ToolBench integration planned Phase 3+)."""

    TASKS = {
        "toolbench-local-1": {
            "query": "Call the weather API for NYC and submit the temperature.",
            "answer": "72F",
            "required_tools": ["list_apis", "call_api", "submit"],
        },
        "toolbench-local-2": {
            "query": "Call the stocks API for AAPL and submit the price.",
            "answer": "190",
            "required_tools": ["list_apis", "call_api", "submit"],
        },
    }

    def __init__(self, budget_tokens: int = 50_000, lambda_penalty: float = 0.001) -> None:
        super().__init__(budget_tokens, lambda_penalty)
        self._task_id = ""
        self._api_called = False
        self._submitted = ""

    def reset(self, task_id: str) -> Observation:
        if task_id not in self.TASKS:
            raise ValueError(f"Unknown ToolBench task: {task_id}")
        self._task_id = task_id
        task = self.TASKS[task_id]
        self._budget = self._budget.__class__(self._budget.budget_tokens)
        self._history = []
        self._turn = 0
        self._done = False
        self._api_called = False
        self._submitted = ""
        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="toolbench",
            query=task["query"],
            budget_tokens=self._budget.budget_tokens,
            tools_available=["list_apis", "call_api", "submit"],
        )
        self._append_history("user", task["query"])
        return self._build_observation("Tool registry loaded. Use list_apis, call_api, or submit.")

    def step(self, action: AgentAction) -> StepResult:
        if self._done:
            return StepResult(obs=self._build_observation("Done."), reward=0.0, done=True)

        self._turn += 1
        tool = action.tool_name
        feedback = ""

        if action.stop or tool == "submit":
            self._submitted = str(action.tool_input or "")
            self._done = True
            feedback = "Submitted."
        elif tool == "list_apis":
            feedback = "Available: weather_api, maps_api"
        elif tool == "call_api":
            self._api_called = True
            if self._task_id == "toolbench-local-2":
                feedback = '{"symbol": "AAPL", "price": "190"}'
            else:
                feedback = '{"temperature": "72F", "city": "NYC"}'
        else:
            feedback = f"Unknown tool: {tool}"

        self._append_history("assistant", action.model_dump_json())
        self._append_history("user", feedback)
        return StepResult(
            obs=self._build_observation(feedback),
            reward=self.compute_step_reward(),
            done=self._done,
            info=StepInfo(step_index=self._turn),
        )

    def evaluate(self) -> EvalResult:
        task = self.TASKS[self._task_id]
        success = self._api_called and (
            self._submitted == task["answer"] or task["answer"] in self._submitted
        )
        return EvalResult(
            success=success,
            final_output=self._submitted,
            total_tokens=self.total_tokens,
            budget_violated=self._budget.violated,
            metadata={"api_called": self._api_called},
        )
