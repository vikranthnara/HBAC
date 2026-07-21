"""Phase 3c: DPO LoRA on tool-JSON capability pairs (TRACE-inspired)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.llm_dpo_trainer import train_dpo_lora

app = typer.Typer(help="Phase 3c: capability-targeted DPO LoRA")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", help="HF model id"),
    lora_rank: int = typer.Option(16, help="LoRA rank"),
    max_pairs: int = typer.Option(400, help="Max DPO pairs"),
    epochs: int = typer.Option(2, help="DPO epochs"),
    beta: float = typer.Option(0.1, help="DPO beta"),
    sft_epochs: int = typer.Option(0, help="SFT warmstart epochs before DPO (0=skip)"),
    reject_modes: str = typer.Option(
        "wrong_tool,invalid_json",
        help="Comma-separated DPO rejection modes",
    ),
    num_batches: int = typer.Option(20, help="Curriculum batches for metadata"),
    output: str = typer.Option("checkpoints/llm_dpo", help="Output root"),
    run_suffix: str = typer.Option("capability", help="Run id suffix"),
    benchmark: str = typer.Option("", help="Filter DPO pairs to one benchmark (empty=all)"),
    oversample_benchmark: str = typer.Option("", help="Oversample pairs from benchmark (e.g. tau_bench)"),
    oversample_factor: int = typer.Option(1, help="Oversample multiplier for benchmark"),
    exclude_benchmarks: str = typer.Option(
        "",
        help="Comma-separated benchmarks to exclude from DPO pairs (holdout policy)",
    ),
    exclude_task_ids_file: str = typer.Option(
        "",
        help="JSON/text file of task IDs to exclude from DPO pairs",
    ),
) -> None:
    root = Path(oracle_path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}_{run_suffix}"
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_curriculum_batches(root, num_batches=num_batches)
    save_batches(batches, out_dir / "batches.jsonl")

    typer.echo(
        f"Phase 3c DPO: pairs<={max_pairs} model={model} beta={beta} sft={sft_epochs} bench={benchmark or 'all'}"
    )
    modes = tuple(m.strip() for m in reject_modes.split(",") if m.strip())
    skip_benches = tuple(b.strip() for b in exclude_benchmarks.split(",") if b.strip())
    excluded_ids: set[str] = set()
    if exclude_task_ids_file:
        raw = Path(exclude_task_ids_file).read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            excluded_ids = set(json.loads(raw))
        else:
            excluded_ids = {ln.strip() for ln in raw.splitlines() if ln.strip()}
    log = train_dpo_lora(
        root,
        model,
        out_dir,
        lora_rank=lora_rank,
        max_pairs=max_pairs,
        epochs=epochs,
        beta=beta,
        sft_epochs=sft_epochs,
        reject_modes=modes or ("wrong_tool", "invalid_json"),
        benchmark=benchmark or None,
        oversample_benchmark=oversample_benchmark or None,
        oversample_factor=oversample_factor,
        exclude_task_ids=excluded_ids or None,
        exclude_benchmarks=skip_benches or None,
    )
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as f:
        for row in log:
            f.write(json.dumps(row) + "\n")
    typer.echo(f"Done -> {out_dir / 'model'}")


if __name__ == "__main__":
    app()
