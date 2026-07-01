"""Phase 3b: LLM GRPO via TRL with oracle-derived prompts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.llm_grpo_trainer import load_sft_prompts, train_with_trl

app = typer.Typer(help="Phase 3b: LLM GRPO (TRL + LoRA)")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    model: str = typer.Option("gpt2", help="HF model id (use Llama-3.1-8B on Rivanna A100)"),
    lora_rank: int = typer.Option(16, help="LoRA rank"),
    grpo_groups: int = typer.Option(8, help="GRPO group size"),
    num_batches: int = typer.Option(10, help="Batch curriculum files"),
    epochs: int = typer.Option(2, help="Training epochs"),
    max_samples: int = typer.Option(64, help="Max SFT prompts"),
    output: str = typer.Option("checkpoints/llm_grpo", help="Output dir"),
) -> None:
    root = Path(oracle_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_curriculum_batches(root, num_batches=num_batches)
    save_batches(batches, out_dir / "batches.jsonl")
    prompts = load_sft_prompts(root, limit=max_samples)
    if not prompts:
        raise typer.Exit("No oracle prompts found for LLM GRPO")

    typer.echo(f"Training on {len(prompts)} oracle prompts, model={model}")
    log = train_with_trl(
        prompts,
        model,
        out_dir,
        lora_rank=lora_rank,
        grpo_groups=grpo_groups,
        epochs=epochs,
        max_samples=max_samples,
    )

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")

    typer.echo(f"Done -> {out_dir / 'model'}")


if __name__ == "__main__":
    app()
