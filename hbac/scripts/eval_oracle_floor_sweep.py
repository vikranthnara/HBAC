"""Oracle floor sweep aligned with live P3 (floors 300-600)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(help="Batch oracle eval across live-matched floors")


@app.command()
def main(
    batches_path: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/batches.jsonl"
    ),
    l2_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"
    ),
    l1_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
    ),
    floors: str = typer.Option("300,400,450,500,600"),
    output: str = typer.Option("results/oracle_floor_sweep.json"),
) -> None:
    sweep: list[dict] = []
    for raw in floors.split(","):
        floor = int(raw.strip())
        out_path = f"results/compose_oracle_floor{floor}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.eval_compose_floor",
                "--batches-path",
                batches_path,
                "--l2-checkpoint",
                l2_checkpoint,
                "--l1-checkpoint",
                l1_checkpoint,
                "--live-min-per-task",
                str(floor),
                "--output",
                out_path,
            ],
            check=True,
        )
        row = json.loads(Path(out_path).read_text())
        sweep.append(
            {
                "floor": floor,
                "file": out_path,
                "hbac_pass_at_1": row.get("hbac_joint", {}).get("pass_at_1"),
                "uniform_pass_at_1": row.get("uniform", {}).get("pass_at_1"),
                "clear_pass_at_1": row.get("clear_compose", {}).get("pass_at_1"),
                "zebra_pass_at_1": row.get("zebra_compose", {}).get("pass_at_1"),
                "hbac_minus_uniform_pp": row.get("hbac_minus_uniform_pp"),
                "hbac_minus_clear_pp": row.get("hbac_minus_clear_pp"),
                "compliant_utility": row.get("metrics", {}),
            }
        )

    report = {"floors": sweep, "budget_fraction": 0.4}
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
