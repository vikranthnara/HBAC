"""Phase 3b v2: SFT warmstart + tool-aware GRPO on oracle step records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.llm_grpo_trainer import train_with_trl

app = typer.Typer(help="Phase 3b v2: tool-aware SFT + GRPO")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HF model id"),
    lora_rank: int = typer.Option(16, help="LoRA rank"),
    grpo_groups: int = typer.Option(8, help="GRPO generations per prompt"),
    grpo_epochs: int = typer.Option(2, help="GRPO epochs"),
    sft_epochs: int = typer.Option(3, help="SFT warmstart epochs (0 to skip)"),
    num_batches: int = typer.Option(20, help="Batch curriculum files"),
    max_samples: int = typer.Option(400, help="Max oracle step records"),
    training_mode: str = typer.Option(
        "sft_then_grpo", help="sft_only | grpo_only | sft_then_grpo"
    ),
    reward_mode: str = typer.Option("tool_aware", help="tool_aware | overlap"),
    max_completion_length: int = typer.Option(384, help="GRPO max completion tokens"),
    output: str = typer.Option("checkpoints/llm_grpo", help="Output dir"),
    run_suffix: str = typer.Option("", help="Optional suffix on run id"),
) -> None:
    root = Path(oracle_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}{'_' + run_suffix if run_suffix else ''}"
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_curriculum_batches(root, num_batches=num_batches)
    save_batches(batches, out_dir / "batches.jsonl")

    typer.echo(
        f"Phase 3b v2: mode={training_mode} reward={reward_mode} "
        f"samples<={max_samples} model={model}"
    )
    log = train_with_trl(
        root,
        model,
        out_dir,
        lora_rank=lora_rank,
        grpo_groups=grpo_groups,
        grpo_epochs=grpo_epochs,
        sft_epochs=sft_epochs,
        max_samples=max_samples,
        training_mode=training_mode,  # type: ignore[arg-type]
        reward_mode=reward_mode,  # type: ignore[arg-type]
        max_completion_length=max_completion_length,
    )

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")

    typer.echo(f"Done -> {out_dir / 'model'}")


if __name__ == "__main__":
    app()
