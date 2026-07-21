from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.llm_grpo_trainer import load_sft_prompts, train_with_trl
from hbac.training.phase3_pipeline import Stage3Config, run_full_phase3

app = typer.Typer(help="Run full Phase 3: Stage 3 L1 GRPO → eval → Stage 4 joint → Variant A")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    checkpoint: str = typer.Option("checkpoints/variant_a", help="L2 checkpoint"),
    output: str = typer.Option("checkpoints/phase3", help="Output root"),
    grpo_groups: int = typer.Option(16, help="L1 GRPO group size"),
    num_batches: int = typer.Option(30, help="Training batches"),
    epochs: int = typer.Option(8, help="Stage 3 epochs"),
    budget_fraction: float = typer.Option(0.90, help="Initial B_total fraction"),
    parse_penalty: float = typer.Option(0.0, help="L1 GRPO parse-failure penalty (D16)"),
    starvation_penalty: float = typer.Option(0.0, help="L1 GRPO starvation penalty (D18)"),
    hard_min_frac: float = typer.Option(0.15, help="Fair min fraction for starvation metric"),
    skip_stage4: bool = typer.Option(False, help="Skip joint Stage 4 even if L1 wins"),
    skip_variant_a: bool = typer.Option(False, help="Skip Variant A utility track"),
    run_llm_grpo: bool = typer.Option(False, help="Also run Phase 3b LLM GRPO"),
    llm_model: str = typer.Option("gpt2", help="HF model for LLM GRPO (gpt2 for local tests)"),
) -> None:
    cfg = Stage3Config(
        grpo_groups=grpo_groups,
        num_batches=num_batches,
        epochs=epochs,
        budget_fraction=budget_fraction,
        parse_penalty=parse_penalty,
        starvation_penalty=starvation_penalty,
        hard_min_frac=hard_min_frac,
    )
    report = run_full_phase3(
        Path(oracle_path),
        Path(checkpoint),
        Path(output),
        cfg,
        run_stage4=not skip_stage4,
        run_variant_a=not skip_variant_a,
    )

    typer.echo("=== Phase 3 Stage 3 ===")
    typer.echo(json.dumps(report.stage3_metrics.to_dict(), indent=2))
    if report.stage4_metrics:
        typer.echo("=== Phase 3 Stage 4 (joint) ===")
        typer.echo(json.dumps(report.stage4_metrics.to_dict(), indent=2))
    else:
        typer.echo("Stage 4 skipped (L1 did not beat uniform or --skip-stage4)")

    if run_llm_grpo:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        llm_dir = Path(output) / "llm_grpo" / run_id
        prompts = load_sft_prompts(Path(oracle_path), limit=100)
        batches = generate_curriculum_batches(Path(oracle_path), num_batches=10)
        save_batches(batches, llm_dir / "batches.jsonl")
        log = train_with_trl(
            prompts,
            llm_model,
            llm_dir,
            grpo_groups=min(grpo_groups, 8),
            epochs=1,
            max_samples=16,
        )
        with (llm_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
            for row in log:
                f.write(json.dumps(row) + "\n")
        typer.echo(f"LLM GRPO -> {llm_dir}")

    typer.echo(f"Report: {report.stage3_dir.parent / 'phase3_report.json'}")


if __name__ == "__main__":
    app()
