"""Audit global_budget vs budget_fraction in training batches.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Audit budget_fraction effectiveness in batches.jsonl")


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl path"),
    output: str = typer.Option("", help="Optional JSON report path"),
) -> None:
    path = Path(batches_path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise typer.Exit("Empty batches file")

    fracs = sorted({r["budget_fraction"] for r in rows})
    budgets = [r["global_budget"] for r in rows]
    oracle_sums = [r["oracle_token_sum"] for r in rows]
    implied = [int(o * r["budget_fraction"]) for r, o in zip(rows, oracle_sums)]
    floors = [len(r["tasks"]) * 40 for r in rows]

    report = {
        "path": str(path),
        "num_batches": len(rows),
        "budget_fraction": fracs[0] if len(fracs) == 1 else fracs,
        "global_budget_unique": sorted(set(budgets)),
        "oracle_sum_range": [min(oracle_sums), max(oracle_sums)],
        "implied_frac_budget_range": [min(implied), max(implied)],
        "floor_40_per_task_range": [min(floors), max(floors)],
        "floor_dominates": sum(1 for b, i in zip(budgets, implied) if b > i),
        "sample": rows[0],
    }

    typer.echo(json.dumps(report, indent=2))
    if output:
        Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    app()
