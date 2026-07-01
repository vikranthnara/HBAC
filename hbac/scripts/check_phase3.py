from __future__ import annotations

from pathlib import Path

import typer

from hbac.gates.phase3_train import phase3a_complete, phase3b_complete, run_phase3_gates
from hbac.gates.report import GateReport, GateStatus

app = typer.Typer(help="Check Phase 3 training completion gates")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    phase3_path: str = typer.Option("checkpoints/phase3", help="Phase 3 output root"),
    l2_checkpoint: str = typer.Option("checkpoints/variant_a", help="L2 checkpoint"),
    output: str = typer.Option("results/phase3_gates.json", help="Report path"),
) -> None:
    report = GateReport(phase1_ready=True, phase2_ready=True, phase3_gateway_ready=True)
    for r in run_phase3_gates(Path(oracle_path), Path(phase3_path), Path(l2_checkpoint)):
        report.add(r)

    report.save(Path(output))
    for line in report.summary_lines():
        typer.echo(line)

    typer.echo(f"Phase 3a complete: {phase3a_complete(report.results)}")
    typer.echo(f"Phase 3b complete: {phase3b_complete(report.results)}")
    typer.echo(f"Phase 4 ready: {phase3a_complete(report.results) and phase3b_complete(report.results)}")

    if not phase3a_complete(report.results):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
