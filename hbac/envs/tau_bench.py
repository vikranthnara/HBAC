"""Minimal τ-bench stub for deterministic CI / Go/No-Go gates."""

from __future__ import annotations

from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec


class TauBenchEnv(BaseAgentEnv):
    """Stub tool–agent–user interaction env (full τ-bench integration planned Phase 3+)."""

    TASKS = {
        "tau-local-1": {
            "query": "Help the user change their flight to tomorrow.",
            "user_goal": "flight_change",
            "answer": "confirmed",
        },
        "tau-local-2": {
            "query": "Help the user cancel their hotel reservation.",
            "user_goal": "hotel_cancel",
            "answer": "cancelled",
        },
        "tau-local-3": {
            "query": "Help the user upgrade their seat on AA200.",
            "user_goal": "seat_upgrade",
            "answer": "upgraded",
        },
    }

    def __init__(self, budget_tokens: int = 50_000, lambda_penalty: float = 0.001) -> None:
        super().__init__(budget_tokens, lambda_penalty)
        self._task_id = ""
        self._lookup_done = False
        self._user_confirmed = False
        self._submitted = ""

    def reset(self, task_id: str) -> Observation:
        if task_id not in self.TASKS:
            raise ValueError(f"Unknown τ-bench task: {task_id}")
        self._task_id = task_id
        task = self.TASKS[task_id]
        self._budget = self._budget.__class__(self._budget.budget_tokens)
        self._history = []
        self._turn = 0
        self._done = False
        self._lookup_done = False
        self._user_confirmed = False
        self._submitted = ""
        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="tau_bench",
            query=task["query"],
            budget_tokens=self._budget.budget_tokens,
            tools_available=["lookup", "message_user", "submit"],
        )
        self._append_history("user", task["query"])
        return self._build_observation(
            "User: I need to change my flight to tomorrow. Use lookup, message_user, submit."
        )

    def step(self, action: AgentAction) -> StepResult:
        if self._done:
            return StepResult(obs=self._build_observation("Done."), reward=0.0, done=True)

        self._turn += 1
        tool = action.tool_name
        feedback = ""

        if action.stop or tool == "submit":
            self._submitted = str(action.tool_input or "")
            self._done = True
            feedback = "Task submitted."
        elif tool == "lookup":
            self._lookup_done = True
            if self._task_id == "tau-local-2":
                feedback = "Hotel reservation H-4421 found; cancellable until tonight."
            elif self._task_id == "tau-local-3":
                feedback = "Seat 12A available on AA200."
            else:
                feedback = "Flight AA100 available tomorrow 9am."
        elif tool == "message_user":
            if self._lookup_done:
                self._user_confirmed = True
                feedback = "User: Yes, please book that flight."
            else:
                feedback = "User: I don't have flight options yet."
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
        success = self._user_confirmed and (
            self._submitted == task["answer"] or task["answer"] in self._submitted
        )
        return EvalResult(
            success=success,
            final_output=self._submitted,
            total_tokens=self.total_tokens,
            budget_violated=self._budget.violated,
            metadata={"user_confirmed": self._user_confirmed},
        )
