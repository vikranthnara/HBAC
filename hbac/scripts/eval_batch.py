from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.training.batch_curriculum import load_batches, sample_batch
from hbac.training.controller import MonolithicController
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.utility_net import UtilityNetwork

app = typer.Typer(help="Evaluate batch Pass@1, budget violations, allocator collapse")


def pass_at_1(results) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def allocation_variance(allocations: dict[str, int]) -> float:
    if not allocations:
        return 0.0
    vals = list(allocations.values())
    return float(np.var(vals))


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl from training"),
    l2_checkpoint: str = typer.Option("checkpoints/variant_a", help="L2 checkpoint dir"),
    allocator: str = typer.Option(
        "auto",
        help="L1 mode: auto|uniform|clear|utility|learned",
    ),
    l1_checkpoint: str | None = typer.Option(None, help="Level1Policy .npz (Variant B)"),
    utility_checkpoint: str | None = typer.Option(None, help="UtilityNetwork .npz (Variant A)"),
    oracle_path: str = typer.Option("data/oracles", help="Oracle root for replay rollouts"),
    output: str = typer.Option("results/batch_eval.json", help="Metrics output"),
) -> None:
    batches = load_batches(Path(batches_path))
    if not batches:
        batches = [sample_batch(Path(oracle_path))]

    ckpts = sorted(
        Path(l2_checkpoint).rglob("stage1_stop_controller.npz"),
        key=lambda p: p.stat().st_mtime,
    )
    l2 = MonolithicController.load(ckpts[-1] if ckpts else Path(l2_checkpoint))

    mode = allocator.lower()
    if mode == "auto":
        if l1_checkpoint:
            mode = "learned"
        elif utility_checkpoint:
            mode = "utility"
        else:
            mode = "uniform"

    l1_policy = Level1Policy.load(Path(l1_checkpoint)) if l1_checkpoint else None
    utility = UtilityNetwork.load(Path(utility_checkpoint)) if utility_checkpoint else None
    clear = CLEARAllocator() if mode == "clear" else None
    oracle_index = OracleIndex(Path(oracle_path))

    all_task_results = []
    batch_violations = 0
    alloc_variances: list[float] = []

    for batch in batches:
        if mode == "learned":
            if not l1_policy:
                raise typer.BadParameter("allocator=learned requires --l1-checkpoint")
            schema_id = int(np.argmax(l1_policy.schema_probs(batch)))
            alloc = l1_policy.allocate_schema(batch, schema_id)
        elif mode == "utility":
            if not utility:
                raise typer.BadParameter("allocator=utility requires --utility-checkpoint")
            alloc = utility.allocate_greedy(batch.tasks, batch.global_budget)
            schema_id = 0
        elif mode == "clear":
            alloc = clear.allocate(batch.tasks, batch.global_budget)
            schema_id = 0
        else:
            alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
            schema_id = 0

        alloc_variances.append(allocation_variance(alloc))
        task_results = []
        for task in batch.tasks:
            task_results.append(
                rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            )
        all_task_results.extend(task_results)
        total_tokens = sum(r.tokens_used for r in task_results)
        if total_tokens > batch.global_budget:
            batch_violations += 1
        for r in task_results:
            if r.tokens_used > r.budget:
                batch_violations += 1

    n_tasks = len(all_task_results)
    metrics = {
        "allocator": mode,
        "pass_at_1": pass_at_1(all_task_results),
        "num_tasks": n_tasks,
        "num_batches": len(batches),
        "batch_budget_violation_rate": batch_violations / max(n_tasks + len(batches), 1),
        "mean_allocation_variance": float(np.mean(alloc_variances)) if alloc_variances else 0.0,
        "allocator_mode_collapse": float(np.mean(alloc_variances) < 1.0),
        "mean_tokens": float(np.mean([r.tokens_used for r in all_task_results])) if all_task_results else 0.0,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
