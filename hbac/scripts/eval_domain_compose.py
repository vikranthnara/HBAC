"""Oracle compose eval on single-benchmark batches (Discovery D4)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.baselines.zebra import ZEBRAAllocator
from hbac.scripts.eval_compose import _eval_clear, _eval_uniform
from hbac.training.batch_curriculum import TrainingBatch, load_batches
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy

app = typer.Typer(help="Domain-filtered oracle compose eval")


def _filter_benchmark(batches: list[TrainingBatch], benchmark: str) -> list[TrainingBatch]:
    out: list[TrainingBatch] = []
    for batch in batches:
        tasks = [t for t in batch.tasks if t.benchmark == benchmark]
        if not tasks:
            continue
        oracle_sum = sum(t.oracle_tokens for t in tasks) or 1
        out.append(
            TrainingBatch(
                batch_id=f"{batch.batch_id}-{benchmark}",
                tasks=tasks,
                global_budget=max(len(tasks) * 40, int(oracle_sum * batch.budget_fraction)),
                oracle_token_sum=oracle_sum,
                budget_fraction=batch.budget_fraction,
            )
        )
    return out


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl"),
    l2_checkpoint: str = typer.Option(...),
    l1_checkpoint: str = typer.Option(...),
    oracle_path: str = typer.Option("data/oracles"),
    benchmark: str = typer.Option("tau_bench", help="Single benchmark filter"),
    output: str = typer.Option("results/domain_compose.json"),
) -> None:
    batches = _filter_benchmark(load_batches(Path(batches_path)), benchmark)
    if not batches:
        raise typer.BadParameter(f"No tasks for benchmark {benchmark}")

    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    clear = CLEARAllocator()
    zebra = ZEBRAAllocator()

    hbac = evaluate_l1_policy(l1, batches, l2, oracle_index).to_dict()
    uniform = _eval_uniform(batches, l2, oracle_index)
    clear_row = _eval_clear(batches, l2, oracle_index, clear)
    zebra_row = _eval_clear(batches, l2, oracle_index, zebra)

    report = {
        "benchmark": benchmark,
        "num_batches": len(batches),
        "num_tasks": hbac.get("num_tasks"),
        "hbac_joint": hbac,
        "uniform": uniform,
        "clear": clear_row,
        "zebra": zebra_row,
        "hbac_minus_uniform_pp": (hbac.get("pass_at_1", 0) - uniform["pass_at_1"]) * 100,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
