from __future__ import annotations

from pathlib import Path

import typer

from hbac.gates.runner import run_all_gates

app = typer.Typer(help="Run Phase 1/2 Go/No-Go gates and Phase 3 gateway checks")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle data root"),
    checkpoint_dir: str = typer.Option("checkpoints/variant_a", help="Variant A checkpoints"),
    output: str = typer.Option("results/go_no_go.json", help="JSON report path"),
    strict: bool = typer.Option(
        True,
        help="Require every gate PASS (exit 1 on FAIL/WARN/BLOCKED)",
    ),
) -> None:
    report = run_all_gates(Path(oracle_path), Path(checkpoint_dir))
    report.save(Path(output))

    for line in report.summary_lines():
        typer.echo(line)

    fails = [r for r in report.results if r.status.value == "fail"]
    blocked = [r for r in report.results if r.status.value == "blocked"]
    warns = [r for r in report.results if r.status.value == "warn"]

    typer.echo(f"\nReport -> {output}")
    typer.echo(
        f"PASS: {sum(1 for r in report.results if r.status.value == 'pass')}/{len(report.results)}  "
        f"FAIL: {len(fails)}  WARN: {len(warns)}  BLOCKED: {len(blocked)}  "
        f"GO Phase 3: {report.go_phase3}"
    )

    if strict and not report.all_pass:
        raise typer.Exit(1)
    if not report.go_phase3:
        raise typer.Exit(2)


if __name__ == "__main__":
    app()
