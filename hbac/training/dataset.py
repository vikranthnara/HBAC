"""Dataset utilities for Variant A stop-controller training."""

from __future__ import annotations

from pathlib import Path

from hbac.core.trajectory import TrajectoryStore
from hbac.core.types import Observation, Trajectory, TrajectoryStep


def find_oracle_paths(root: Path) -> list[Path]:
    """Find all oracles.jsonl under a directory tree, sorted by path."""
    if root.is_file():
        return [root]
    return sorted(root.rglob("oracles.jsonl"))


def find_all_trajectory_paths(root: Path) -> list[Path]:
    """Find all_trajectories.jsonl files for negative-example mining."""
    if root.is_file():
        parent = root.parent / "all_trajectories.jsonl"
        return [parent] if parent.is_file() else []
    return sorted(root.rglob("all_trajectories.jsonl"))


def load_training_subset(
    path: Path,
    *,
    limit: int | None = None,
    successful_only: bool = True,
) -> list[Trajectory]:
    store = TrajectoryStore(path)
    trajs = store.load_successful() if successful_only else store.load_all()
    if limit:
        trajs = trajs[:limit]
    return trajs


def _build_history(traj: Trajectory, step_index: int) -> list[dict[str, str]]:
    """Reconstruct conversation history up to (not including) step_index."""
    history: list[dict[str, str]] = []
    for j, prior in enumerate(traj.steps[:step_index]):
        if prior.llm_response:
            history.append({"role": "assistant", "content": prior.llm_response})
        if prior.observation:
            history.append({"role": "user", "content": prior.observation})
    return history


def _step_observation(traj: Trajectory, step_index: int, step: TrajectoryStep) -> Observation:
    tokens_used = sum(s.tokens for s in traj.steps[:step_index])
    return Observation(
        history=_build_history(traj, step_index),
        env_feedback=step.observation,
        turn=step.turn,
        remaining_budget=max(0, traj.budget - tokens_used),
        tools_available=[],
    )


def trajectory_to_stop_examples(traj: Trajectory, *, successful_only: bool = True) -> list[dict]:
    """Convert trajectory steps to (observation, stop_label) pairs for Stage 1."""
    if successful_only and not traj.success:
        return []

    examples: list[dict] = []
    for i, step in enumerate(traj.steps):
        obs = _step_observation(traj, i, step)
        is_last = i == len(traj.steps) - 1
        # Oracle stop: only on final step of successful trajectories
        stop_label = bool(traj.success and is_last)
        cumulative_tokens = sum(s.tokens for s in traj.steps[: i + 1])
        examples.append(
            {
                "observation": obs,
                "stop": stop_label,
                "tokens": cumulative_tokens,
                "success": traj.success,
                "task_id": traj.task_id,
                "turn": step.turn,
            }
        )
    return examples


def failed_trajectory_negatives(traj: Trajectory) -> list[dict]:
    """Non-terminal steps from failed trajectories: premature stop should be penalized."""
    if traj.success or not traj.steps:
        return []

    examples: list[dict] = []
    for i, step in enumerate(traj.steps[:-1]):
        obs = _step_observation(traj, i, step)
        cumulative_tokens = sum(s.tokens for s in traj.steps[: i + 1])
        examples.append(
            {
                "observation": obs,
                "stop": False,
                "tokens": cumulative_tokens,
                "success": False,
                "task_id": traj.task_id,
                "turn": step.turn,
                "negative": True,
            }
        )
    return examples


def load_stop_examples(
    paths: list[Path],
    *,
    limit: int | None = None,
    include_negatives: bool = True,
    negative_root: Path | None = None,
) -> list[dict]:
    """Load stop examples from one or more oracles.jsonl files."""
    examples: list[dict] = []
    seen_oracle = set()

    for path in paths:
        key = str(path.resolve())
        if key in seen_oracle:
            continue
        seen_oracle.add(key)
        for traj in load_training_subset(path, successful_only=True):
            examples.extend(trajectory_to_stop_examples(traj))
            if limit and len(examples) >= limit:
                return examples[:limit]

    if include_negatives and negative_root is not None:
        for all_path in find_all_trajectory_paths(negative_root):
            for traj in load_training_subset(all_path, successful_only=False):
                examples.extend(failed_trajectory_negatives(traj))
                if limit and len(examples) >= limit:
                    return examples[:limit]

    return examples[:limit] if limit else examples


def train_val_split(
    examples: list[dict],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split examples by task_id to avoid leakage."""
    import random

    by_task: dict[str, list[dict]] = {}
    for ex in examples:
        by_task.setdefault(ex["task_id"], []).append(ex)

    task_ids = sorted(by_task.keys())
    rng = random.Random(seed)
    rng.shuffle(task_ids)
    n_val = max(1, int(len(task_ids) * val_fraction)) if len(task_ids) > 1 else 0
    val_tasks = set(task_ids[:n_val])

    train, val = [], []
    for tid, exs in by_task.items():
        (val if tid in val_tasks else train).extend(exs)
    return train, val
