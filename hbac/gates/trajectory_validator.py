from __future__ import annotations

import json
import re

from hbac.core.types import AgentAction, Trajectory


def validate_action_parse(llm_response: str) -> tuple[bool, str]:
    """Return (ok, reason) for JSON tool-call parse."""
    text = (llm_response or "").strip()
    if not text:
        return False, "empty_response"
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return False, "no_json_object"
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return False, "invalid_json"
    if not isinstance(data, dict):
        return False, "not_object"
    tool = data.get("tool_name", data.get("action"))
    if not tool:
        return False, "missing_tool_name"
    return True, "ok"


def validate_trajectory_pomdp(traj: Trajectory) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors: list[str] = []
    if not traj.task_id:
        errors.append("missing_task_id")
    if not traj.benchmark:
        errors.append("missing_benchmark")
    if traj.budget <= 0:
        errors.append("invalid_budget")
    if not traj.steps:
        errors.append("no_steps")
        return errors

    cumulative = 0
    for i, step in enumerate(traj.steps):
        if step.turn != i and step.turn != traj.steps[i - 1].turn + 1 if i else step.turn != 0:
            if step.turn < 0:
                errors.append(f"step_{i}_bad_turn")
        if step.tokens < 0:
            errors.append(f"step_{i}_negative_tokens")
        cumulative += step.tokens
        if not step.action.tool_name:
            errors.append(f"step_{i}_missing_tool")
        ok, reason = validate_action_parse(step.llm_response)
        if step.llm_response and not ok:
            errors.append(f"step_{i}_parse_{reason}")

    if traj.success and traj.total_tokens <= 0 and cumulative <= 0:
        errors.append("success_with_zero_tokens")

    return errors


def pomdp_compliance_rate(trajectories: list[Trajectory]) -> tuple[float, list[str]]:
    if not trajectories:
        return 0.0, ["no_trajectories"]
    bad = []
    for t in trajectories:
        errs = validate_trajectory_pomdp(t)
        if errs:
            bad.append(f"{t.task_id}:{'|'.join(errs)}")
    rate = 1.0 - len(bad) / len(trajectories)
    return rate, bad
