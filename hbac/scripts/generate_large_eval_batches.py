"""Generate large held-out batch files for n>=1000 live eval."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches

app = typer.Typer(help="Generate eval batches at scale (e.g. 200 batches ≈ 1200 tasks)")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles/real_eval/latest", help="Oracle root"),
    num_batches: int = typer.Option(200, help="Number of batches (~6 tasks each)"),
    seed: int = typer.Option(47, help="RNG seed"),
    output: str = typer.Option("checkpoints/eval_n1000/batches.jsonl"),
) -> None:
    root = Path(oracle_path)
    if root.is_symlink():
        root = root.parent / root.readlink()
    if not root.is_dir():
        root = Path("data/oracles")
    batches = generate_curriculum_batches(root, num_batches=num_batches, seed=seed)
    out = Path(output)
    save_batches(batches, out)
    n_tasks = sum(len(b.tasks) for b in batches)
    meta = {
        "num_batches": len(batches),
        "num_tasks": n_tasks,
        "oracle_path": str(root),
        "seed": seed,
        "output": str(out),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    typer.echo(json.dumps(meta, indent=2))


if __name__ == "__main__":
    app()
