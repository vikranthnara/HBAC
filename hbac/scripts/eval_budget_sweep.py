"""Oracle-replay budget fraction sweep: where does HBAC pass@1 gap peak? (Discovery D1)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.scripts.eval_compose import _eval_clear, _eval_uniform
from hbac.training.batch_curriculum import load_batches, sample_batch, save_batches
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy

app = typer.Typer(help="Oracle compose eval across budget fractions")


@app.command()
def main(
    l2_checkpoint: str = typer.Option(..., help="Frozen L2 .npz or dir"),
    l1_checkpoint: str = typer.Option(..., help="HBAC L1 .npz"),
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    fractions: str = typer.Option("0.20,0.25,0.30,0.35,0.40,0.45,0.50", help="Budget fractions"),
    batches_per_frac: int = typer.Option(30, help="Batches sampled per fraction"),
    seed: int = typer.Option(47, help="RNG seed"),
    output: str = typer.Option("results/budget_sweep_oracle.json", help="Output JSON"),
) -> None:
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    clear = CLEARAllocator()

    sweep: list[dict] = []
    for raw in fractions.split(","):
        frac = float(raw.strip())
        batches = [
            sample_batch(Path(oracle_path), budget_fraction=frac, seed=seed + i)
            for i in range(batches_per_frac)
        ]
        hbac = evaluate_l1_policy(l1, batches, l2, oracle_index).to_dict()
        uniform = _eval_uniform(batches, l2, oracle_index)
        clear_row = _eval_clear(batches, l2, oracle_index, clear)
        sweep.append(
            {
                "budget_fraction": frac,
                "num_batches": len(batches),
                "num_tasks": hbac.get("num_tasks"),
                "hbac_pass_at_1": hbac.get("pass_at_1"),
                "uniform_pass_at_1": uniform["pass_at_1"],
                "clear_pass_at_1": clear_row["pass_at_1"],
                "hbac_minus_uniform_pp": (hbac.get("pass_at_1", 0) - uniform["pass_at_1"]) * 100,
                "hbac_mean_reward": hbac.get("mean_batch_reward"),
                "uniform_mean_reward": uniform["mean_batch_reward"],
                "hbac_allocation_variance": hbac.get("mean_allocation_variance"),
            }
        )

    best = max(sweep, key=lambda r: r["hbac_minus_uniform_pp"])
    report = {
        "fractions": sweep,
        "peak_gap_pp": best["hbac_minus_uniform_pp"],
        "peak_fraction": best["budget_fraction"],
        "discovery_note": (
            "If gap increases as fraction decreases, prioritize live eval at lowest fraction "
            "where oracle gap is maximal."
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
