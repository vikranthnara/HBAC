"""Build step-level GRPO/SFT records from oracle trajectories (matches live eval prompts)."""

from __future__ import annotations

from pathlib import Path

from hbac.core.trajectory import TrajectoryStore
from hbac.training.dataset import find_oracle_paths
from hbac.training.tool_reward import build_chat_prompt, format_prompt_for_trl


def _history_before(traj, step_index: int) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for j, prior in enumerate(traj.steps[:step_index]):
        if prior.llm_response:
            history.append({"role": "assistant", "content": prior.llm_response})
        if prior.observation:
            history.append({"role": "user", "content": prior.observation})
    return history


def load_grpo_step_records(
    oracle_root: Path,
    *,
    limit: int = 500,
    successful_only: bool = True,
    benchmark: str | None = None,
) -> list[dict]:
    """One record per oracle step with chat messages + reference tool metadata."""
    records: list[dict] = []
    for path in find_oracle_paths(oracle_root):
        trajs = TrajectoryStore(path).load_successful() if successful_only else TrajectoryStore(path).load_all()
        for traj in trajs:
            if successful_only and not traj.success:
                continue
            if benchmark and traj.benchmark != benchmark:
                continue
            for i, step in enumerate(traj.steps):
                if not step.llm_response:
                    continue
                user = step.observation or f"Task {traj.task_id} turn {step.turn}"
                messages = build_chat_prompt(
                    traj.benchmark,
                    user_content=user,
                    history=_history_before(traj, i),
                )
                ref_tool = step.action.tool_name if step.action else None
                records.append(
                    {
                        "messages": messages,
                        "completion": step.llm_response,
                        "reference_tool": ref_tool,
                        "task_id": traj.task_id,
                        "benchmark": traj.benchmark,
                        "turn": step.turn,
                        "success": traj.success,
                        "reward_weight": 1.0 if traj.success else 0.3,
                    }
                )
                if len(records) >= limit:
                    return records
    return records


def records_to_trl_prompts(records: list[dict], tokenizer) -> list[dict]:
    out: list[dict] = []
    for row in records:
        prompt = format_prompt_for_trl(row["messages"], tokenizer)
        out.append({**row, "prompt": prompt})
    return out
