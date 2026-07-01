"""Compose-vs-joint evaluation with live LLM rollouts (not oracle replay)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator, allocation_variance
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend
from hbac.dotenv_loader import load_project_env
from hbac.training.batch_curriculum import TrainingBatch, load_batches
from hbac.training.batch_rollout import rollout_task
from hbac.training.controller import MonolithicController
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.phase3_pipeline import _resolve_l2
from hbac.training.reward import TaskControllerReward

app = typer.Typer(help="Live-LLM compose eval: uniform vs CLEAR vs HBAC")

STUB_BENCHMARKS = frozenset({"tau_bench", "toolbench", "mock", "swe_bench"})
LIVE_MIN_PER_TASK = 400


def _filter_batches(
    batches: list[TrainingBatch],
    *,
    benchmarks: set[str] | None,
    max_batches: int | None,
    live: bool = True,
) -> list[TrainingBatch]:
    out: list[TrainingBatch] = []
    for batch in batches:
        tasks = [t for t in batch.tasks if not benchmarks or t.benchmark in benchmarks]
        if not tasks:
            continue
        oracle_sum = sum(t.oracle_tokens for t in tasks) or 1
        n = len(tasks)
        frac_budget = int(oracle_sum * batch.budget_fraction)
        floor = n * (LIVE_MIN_PER_TASK if live else 40)
        out.append(
            TrainingBatch(
                batch_id=batch.batch_id,
                tasks=tasks,
                global_budget=max(floor, frac_budget),
                oracle_token_sum=oracle_sum,
                budget_fraction=batch.budget_fraction,
            )
        )
        if max_batches and len(out) >= max_batches:
            break
    return out


def _eval_allocator(
    name: str,
    batches: list[TrainingBatch],
    l2: MonolithicController,
    llm: LLMBackend,
    alloc_fn,
) -> dict:
    reward_fn = TaskControllerReward()
    rewards: list[float] = []
    successes: list[bool] = []
    violations = 0
    alloc_vars: list[float] = []

    for batch in batches:
        alloc = alloc_fn(batch)
        alloc_vars.append(allocation_variance(alloc))
        results = []
        for task in batch.tasks:
            r = rollout_task(task, alloc[task.task_id], l2, reward_fn, llm=llm)
            results.append(r)
            successes.append(r.success)
            if r.budget_violated:
                violations += 1
        rewards.append(l1_schema_reward(results, batch, alloc))
        if sum(r.tokens_used for r in results) > batch.global_budget:
            violations += 1

    n = max(len(successes), 1)
    return {
        "allocator": name,
        "pass_at_1": sum(successes) / n,
        "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
        "batch_violation_rate": violations / (n + len(batches)),
        "mean_allocation_variance": float(np.mean(alloc_vars)) if alloc_vars else 0.0,
        "num_tasks": len(successes),
        "num_batches": len(batches),
    }


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl from training run"),
    l2_checkpoint: str = typer.Option(..., help="Frozen L2 checkpoint"),
    l1_checkpoint: str = typer.Option(..., help="Learned HBAC L1 .npz"),
    output: str = typer.Option("results/compose_live.json", help="Metrics output"),
    llm_spec: str = typer.Option("auto", help="LLM spec: auto | provider:model"),
    benchmarks: str = typer.Option(
        "tau_bench,toolbench,mock",
        help="Comma-separated benchmarks (live eval; LCB needs oracle replay)",
    ),
    max_batches: int = typer.Option(10, help="Cap batches for API cost control"),
) -> None:
    load_project_env()
    bench_set = {b.strip() for b in benchmarks.split(",") if b.strip()}
    batches = _filter_batches(
        load_batches(Path(batches_path)),
        benchmarks=bench_set,
        max_batches=max_batches,
    )
    if not batches:
        raise typer.BadParameter(f"No batches with benchmarks {bench_set}")

    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    llm = LLMBackend.from_spec(llm_spec) if llm_spec != "auto" else LLMBackend.from_config(LLMConfig())
    clear = CLEARAllocator()

    typer.echo(f"Live LLM: {llm.config.provider}:{llm.config.model}")
    typer.echo(f"Batches: {len(batches)} tasks: {sum(len(b.tasks) for b in batches)}")

    uniform = _eval_allocator(
        "uniform",
        batches,
        l2,
        llm,
        lambda b: Level1Allocator(b.global_budget).allocate(b.task_ids),
    )
    clear_compose = _eval_allocator(
        "clear",
        batches,
        l2,
        llm,
        lambda b: clear.allocate(b.tasks, b.global_budget),
    )

    sid = int(np.argmax(l1.schema_probs(batches[0])))
    hbac = _eval_allocator(
        "hbac_joint",
        batches,
        l2,
        llm,
        lambda b: l1.allocate_schema(b, int(np.argmax(l1.schema_probs(b)))),
    )

    report = {
        "llm": f"{llm.config.provider}:{llm.config.model}",
        "benchmarks": sorted(bench_set),
        "budget_fraction": batches[0].budget_fraction if batches else None,
        "uniform": uniform,
        "clear_compose": clear_compose,
        "hbac_joint": hbac,
        "hbac_beats_clear": hbac["pass_at_1"] > clear_compose["pass_at_1"]
        or hbac["mean_batch_reward"] > clear_compose["mean_batch_reward"],
        "hbac_beats_uniform": hbac["pass_at_1"] > uniform["pass_at_1"]
        or hbac["mean_batch_reward"] > uniform["mean_batch_reward"],
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
