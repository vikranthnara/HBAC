"""Oracle compose eval with live-matched per-task floors (P3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator, allocation_variance
from hbac.baselines.zebra import ZEBRAAllocator
from hbac.eval.batch_floor import load_batches_with_floor
from hbac.scripts.eval_compose import _eval_clear, _eval_uniform
from hbac.training.level1 import Level1Policy
from hbac.training.metrics import summarize_allocator_row
from hbac.training.oracle_replay import OracleIndex
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy

app = typer.Typer(help="Oracle eval with live-style token floors (P3)")


@app.command()
def main(
    batches_path: str = typer.Option(...),
    l2_checkpoint: str = typer.Option(...),
    l1_checkpoint: str = typer.Option(...),
    oracle_path: str = typer.Option("data/oracles"),
    live_min_per_task: int = typer.Option(400, help="Match live eval floor"),
    budget_fraction: float = typer.Option(0.4),
    output: str = typer.Option("results/compose_oracle_floor400.json"),
) -> None:
    batches = load_batches_with_floor(
        Path(batches_path),
        live_min_per_task=live_min_per_task,
        budget_fraction=budget_fraction,
    )
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
        "live_min_per_task": live_min_per_task,
        "budget_fraction": budget_fraction,
        "num_batches": len(batches),
        "hbac_joint": hbac,
        "uniform": uniform,
        "clear_compose": clear_row,
        "zebra_compose": zebra_row,
        "metrics": {
            "hbac": summarize_allocator_row(hbac),
            "uniform": summarize_allocator_row(uniform),
            "clear": summarize_allocator_row(clear_row),
            "zebra": summarize_allocator_row(zebra_row),
        },
        "hbac_minus_uniform_pp": (hbac.get("pass_at_1", 0) - uniform["pass_at_1"]) * 100,
        "hbac_minus_clear_pp": (hbac.get("pass_at_1", 0) - clear_row["pass_at_1"]) * 100,
    }
    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
