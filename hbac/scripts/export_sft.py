from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.core.trajectory import TrajectoryStore

app = typer.Typer(help="Export oracle trajectories to SFT and GRPO init formats")


def export_sft(trajectories_path: Path, output_path: Path) -> int:
    store = TrajectoryStore(trajectories_path)
    trajectories = store.load_successful()
    records = []

    for traj in trajectories:
        messages = []
        for step in traj.steps:
            if step.llm_response:
                messages.append({"role": "assistant", "content": step.llm_response})
            if step.observation:
                messages.append({"role": "user", "content": step.observation})

        records.append(
            {
                "task_id": traj.task_id,
                "benchmark": traj.benchmark,
                "messages": messages,
                "labels": {
                    "budget_allocations": [s.budget_allocated for s in traj.steps],
                    "stop_labels": [s.action.stop for s in traj.steps],
                    "success": traj.success,
                },
                "metadata": traj.metadata,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def export_grpo_groups(trajectories_path: Path, output_path: Path, group_size: int = 4) -> int:
    store = TrajectoryStore(trajectories_path)
    trajectories = store.load_all()

    by_task: dict[str, list] = {}
    for traj in trajectories:
        by_task.setdefault(traj.task_id, []).append(traj)

    groups = []
    for task_id, trajs in by_task.items():
        reward = 1.0 if any(t.success for t in trajs) else 0.0
        groups.append(
            {
                "task_id": task_id,
                "benchmark": trajs[0].benchmark,
                "group": [t.model_dump() for t in trajs[:group_size]],
                "rewards": [1.0 if t.success else 0.0 for t in trajs[:group_size]],
                "group_reward": reward,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g) + "\n")
    return len(groups)


@app.command()
def main(
    input_path: str = typer.Option(..., help="Path to trajectories JSONL"),
    output_dir: str = typer.Option("data/training", help="Output directory"),
    format: str = typer.Option("both", help="sft | grpo | both"),
    group_size: int = typer.Option(4, help="GRPO group size"),
) -> None:
    in_path = Path(input_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if format in {"sft", "both"}:
        n = export_sft(in_path, out / "sft.jsonl")
        typer.echo(f"Exported {n} SFT records -> {out / 'sft.jsonl'}")

    if format in {"grpo", "both"}:
        n = export_grpo_groups(in_path, out / "grpo_groups.jsonl", group_size=group_size)
        typer.echo(f"Exported {n} GRPO groups -> {out / 'grpo_groups.jsonl'}")


if __name__ == "__main__":
    app()
