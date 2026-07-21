"""CLEAR baseline sensitivity sweep on oracle replay (W5 mitigation)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.scripts.eval_compose import _eval_clear, _eval_uniform
from hbac.training.batch_curriculum import load_batches
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy

app = typer.Typer(help="CLEAR proxy sensitivity on oracle compose eval")


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl"),
    l2_checkpoint: str = typer.Option(..., help="Frozen L2 checkpoint"),
    l1_checkpoint: str = typer.Option(..., help="HBAC L1 .npz"),
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    min_per_task_values: str = typer.Option("50,100,200,400", help="CLEAR min_per_task sweep"),
    output: str = typer.Option("results/clear_sensitivity.json", help="Output JSON"),
) -> None:
    batches = load_batches(Path(batches_path))
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    hbac = evaluate_l1_policy(l1, batches, l2, oracle_index).to_dict()
    uniform = _eval_uniform(batches, l2, oracle_index)

    sweep: list[dict] = []
    for raw in min_per_task_values.split(","):
        min_pt = int(raw.strip())
        clear = CLEARAllocator(min_per_task=min_pt)
        row = _eval_clear(batches, l2, oracle_index, clear)
        sweep.append(
            {
                "min_per_task": min_pt,
                "pass_at_1": row["pass_at_1"],
                "mean_batch_reward": row["mean_batch_reward"],
                "mean_allocation_variance": row["mean_allocation_variance"],
                "beats_uniform": row["pass_at_1"] > uniform["pass_at_1"],
            }
        )

    report = {
        "batches_path": batches_path,
        "num_tasks": hbac.get("num_tasks"),
        "hbac_joint": {"pass_at_1": hbac.get("pass_at_1"), "mean_batch_reward": hbac.get("mean_batch_reward")},
        "uniform": {"pass_at_1": uniform["pass_at_1"], "mean_batch_reward": uniform["mean_batch_reward"]},
        "clear_sweep": sweep,
        "clear_ever_beats_uniform": any(r["beats_uniform"] for r in sweep),
        "hbac_beats_clear_all_settings": all(
            hbac.get("pass_at_1", 0) > r["pass_at_1"] for r in sweep
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
