"""Phase 3c: TRACE-style capability deficit analysis."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.training.capability import build_dpo_pairs, write_capability_report

app = typer.Typer(help="Analyze capability deficits and export DPO pair preview")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    output: str = typer.Option("results/capability_report.json", help="Deficit report JSON"),
    dpo_preview: str = typer.Option("results/dpo_pairs_preview.jsonl", help="Sample DPO pairs"),
    pair_limit: int = typer.Option(50, help="Max DPO pairs in preview"),
) -> None:
    root = Path(oracle_path)
    report = write_capability_report(root, Path(output))
    typer.echo(json.dumps(report, indent=2))

    pairs = build_dpo_pairs(root, limit=pair_limit)
    preview_path = Path(dpo_preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    with preview_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(
                json.dumps(
                    {
                        "capability_id": p.capability_id,
                        "benchmark": p.benchmark,
                        "task_id": p.task_id,
                        "chosen_reward": p.chosen_reward,
                        "rejected_reward": p.rejected_reward,
                        "prompt_chars": len(p.prompt),
                    }
                )
                + "\n"
            )
    typer.echo(f"Wrote {len(pairs)} DPO pair previews -> {preview_path}")


if __name__ == "__main__":
    app()
