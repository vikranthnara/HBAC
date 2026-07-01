from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_id: str
    benchmark: str
    query: str
    budget_tokens: int
    tools_available: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    history: list[dict[str, str]] = Field(default_factory=list)
    env_feedback: str = ""
    tools_available: list[str] = Field(default_factory=list)
    draft_signals: dict[str, float] | None = None
    turn: int = 0
    remaining_budget: int = 0


class AgentAction(BaseModel):
    thought: str | None = None
    tool_name: str
    tool_input: str | dict[str, Any] | None = None
    stop: bool = False
    max_tokens: int | None = None


class StepInfo(BaseModel):
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_cost: int = 0
    step_index: int = 0
    budget_allocated: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    obs: Observation
    reward: float
    done: bool
    info: StepInfo = Field(default_factory=StepInfo)


class EvalResult(BaseModel):
    success: bool
    final_output: str = ""
    test_output: str = ""
    total_tokens: int = 0
    budget_violated: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    turn: int
    state_summary: str = ""
    action: AgentAction
    observation: str = ""
    tokens: int = 0
    budget_allocated: int | None = None
    llm_response: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    task_id: str
    benchmark: str
    model: str
    baseline: str
    success: bool
    total_tokens: int
    budget: int
    steps: list[TrajectoryStep] = Field(default_factory=list)
    final_output: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
