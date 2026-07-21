"""Build DPO holdout policy and audit manifest (exclude eval benchmark families)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import typer

from hbac.training.batch_curriculum import load_batches
from hbac.training.capability import build_dpo_pairs

app = typer.Typer(help="Prepare DPO holdout policy excluding eval benchmark overlap")


def _sha256_sorted(ids: list[str]) -> str:
    payload = "\n".join(sorted(set(ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.command()
def main(
    oracle_root: str = typer.Option("data/oracles", help="DPO pair source oracles"),
    eval_batches: str = typer.Option(
        "checkpoints/eval_n1000/batches.jsonl",
        help="V3 live eval batches",
    ),
    dpo_limit: int = typer.Option(600, help="Max DPO pairs"),
    output: str = typer.Option("results/dpo_holdout_policy.json"),
    retrain_script: str = typer.Option("slurm/train_llm_dpo_holdout.sh"),
) -> None:
    pairs_full = build_dpo_pairs(Path(oracle_root), limit=dpo_limit, reject_modes=("wrong_tool",))
    batches = load_batches(Path(eval_batches))
    eval_by_bench: dict[str, set[str]] = defaultdict(set)
    for batch in batches:
        for task in batch.tasks:
            eval_by_bench[task.benchmark].add(task.task_id)

    train_by_bench: dict[str, set[str]] = defaultdict(set)
    for p in pairs_full:
        train_by_bench[p.benchmark].add(p.task_id)

    # Exclude only benchmark families with exact task-ID overlap (plan §5B).
    exclude_benchmarks = tuple(
        sorted(
            bench
            for bench in set(train_by_bench) & set(eval_by_bench)
            if train_by_bench[bench] & eval_by_bench[bench]
        )
    )
    pairs_holdout = build_dpo_pairs(
        Path(oracle_root),
        limit=dpo_limit,
        reject_modes=("wrong_tool",),
        exclude_benchmarks=exclude_benchmarks,
    )

    eval_ids = [tid for ids in eval_by_bench.values() for tid in ids]
    train_ids = [p.task_id for p in pairs_holdout]
    overlap = set(train_ids) & set(eval_ids)

    report = {
        "policy": "exclude_eval_benchmark_families",
        "exclude_benchmarks": list(exclude_benchmarks),
        "eval_batches": eval_batches,
        "oracle_root": oracle_root,
        "dpo_pairs_before": len(pairs_full),
        "dpo_pairs_after": len(pairs_holdout),
        "exact_task_overlap": len(overlap),
        "overlap_task_ids": sorted(overlap),
        "manifest": {
            "holdout_train_task_ids_sha256": _sha256_sorted(train_ids),
            "eval_task_ids_sha256": _sha256_sorted(eval_ids),
        },
        "verdict": "PASS" if len(overlap) == 0 else "FAIL",
        "retrain_command": (
            f"python -m hbac.scripts.train_llm_dpo "
            f"--exclude-benchmarks {','.join(exclude_benchmarks)} "
            f"--run-suffix capability_holdout"
        ),
        "retrain_slurm": retrain_script,
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
