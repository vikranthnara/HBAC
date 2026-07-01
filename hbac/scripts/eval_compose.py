"""Compose-vs-joint baseline: CLEAR + frozen L2 vs learned HBAC L1+L2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator, allocation_variance
from hbac.training.batch_curriculum import load_batches
from hbac.training.controller import MonolithicController
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy
from hbac.training.l1_batch_reward import l1_schema_reward

app = typer.Typer(help="Compare CLEAR compose vs HBAC joint vs uniform")


def _pass_at_1(results) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def _eval_clear(
    batches,
    l2: MonolithicController,
    oracle_index: OracleIndex,
    allocator: CLEARAllocator,
) -> dict:
    rewards: list[float] = []
    successes: list[bool] = []
    violations = 0
    alloc_vars: list[float] = []

    for batch in batches:
        alloc = allocator.allocate(batch.tasks, batch.global_budget)
        alloc_vars.append(allocation_variance(alloc))
        results = [
            rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        successes.extend(r.success for r in results)
        rewards.append(l1_schema_reward(results, batch, alloc))
        violations += sum(1 for r in results if r.budget_violated)
        total = sum(r.tokens_used for r in results)
        if total > batch.global_budget:
            violations += 1

    n = max(len(successes), 1)
    return {
        "pass_at_1": sum(successes) / n,
        "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
        "batch_violation_rate": violations / (n + len(batches)),
        "mean_allocation_variance": float(np.mean(alloc_vars)) if alloc_vars else 0.0,
        "num_tasks": len(successes),
        "num_batches": len(batches),
    }


def _eval_uniform(batches, l2, oracle_index) -> dict:
    rewards: list[float] = []
    successes: list[bool] = []
    violations = 0
    alloc_vars: list[float] = []

    for batch in batches:
        alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        alloc_vars.append(allocation_variance(alloc))
        results = [
            rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        successes.extend(r.success for r in results)
        rewards.append(l1_schema_reward(results, batch, alloc))
        violations += sum(1 for r in results if r.budget_violated)

    n = max(len(successes), 1)
    return {
        "pass_at_1": sum(successes) / n,
        "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
        "batch_violation_rate": violations / (n + len(batches)),
        "mean_allocation_variance": float(np.mean(alloc_vars)) if alloc_vars else 0.0,
        "num_tasks": len(successes),
        "num_batches": len(batches),
    }


@app.command()
def main(
    batches_path: str = typer.Option(..., help="Held-out batches.jsonl"),
    l2_checkpoint: str = typer.Option("checkpoints/variant_a", help="Frozen L2 checkpoint dir"),
    l1_checkpoint: str = typer.Option(..., help="Learned HBAC L1 .npz (joint compose target)"),
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    output: str = typer.Option("results/compose_vs_joint.json", help="Metrics output"),
) -> None:
    batches = load_batches(Path(batches_path))
    if not batches:
        raise typer.BadParameter(f"No batches in {batches_path}")

    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    clear = CLEARAllocator()

    hbac_joint = evaluate_l1_policy(l1, batches, l2, oracle_index)
    clear_compose = _eval_clear(batches, l2, oracle_index, clear)
    uniform = _eval_uniform(batches, l2, oracle_index)

    report = {
        "uniform": uniform,
        "clear_compose": clear_compose,
        "hbac_joint": hbac_joint.to_dict(),
        "clear_beats_uniform": clear_compose["pass_at_1"] > uniform["pass_at_1"],
        "hbac_beats_clear": hbac_joint.pass_at_1 > clear_compose["pass_at_1"],
        "hbac_beats_uniform": hbac_joint.beats_uniform,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
