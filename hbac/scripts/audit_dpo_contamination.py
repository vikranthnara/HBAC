"""Audit DPO training pairs vs eval task IDs for leakage."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import typer

from hbac.training.batch_curriculum import load_batches
from hbac.training.capability import build_dpo_pairs

app = typer.Typer(help="DPO contamination audit with SHA256 manifests")


def _sha256_sorted(ids: list[str]) -> str:
    payload = "\n".join(sorted(set(ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@app.command()
def main(
    oracle_root: str = typer.Option("data/oracles", help="DPO pair source oracles"),
    dpo_limit: int = typer.Option(600, help="Max DPO pairs"),
    eval_batches: str = typer.Option(
        "checkpoints/eval_n1000/batches.jsonl",
        help="V3 live eval batches",
    ),
    exclude_benchmarks: str = typer.Option(
        "",
        help="Comma-separated benchmarks excluded from DPO (holdout policy audit)",
    ),
    output: str = typer.Option("results/dpo_contamination_audit.json"),
) -> None:
    skip_benches = tuple(b.strip() for b in exclude_benchmarks.split(",") if b.strip())
    pairs = build_dpo_pairs(
        Path(oracle_root),
        limit=dpo_limit,
        reject_modes=("wrong_tool",),
        exclude_benchmarks=skip_benches or None,
    )
    train_ids = [p.task_id for p in pairs]
    train_by_bench: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        train_by_bench[p.benchmark].add(p.task_id)

    batches = load_batches(Path(eval_batches))
    eval_ids: list[str] = []
    eval_by_bench: dict[str, set[str]] = defaultdict(set)
    for batch in batches:
        for task in batch.tasks:
            eval_ids.append(task.task_id)
            eval_by_bench[task.benchmark].add(task.task_id)

    train_set = set(train_ids)
    eval_set = set(eval_ids)
    overlap = train_set & eval_set

    per_bench = []
    for bench in sorted(set(train_by_bench) | set(eval_by_bench)):
        t = train_by_bench.get(bench, set())
        e = eval_by_bench.get(bench, set())
        inter = t & e
        per_bench.append(
            {
                "benchmark": bench,
                "train_tasks": len(t),
                "eval_tasks": len(e),
                "exact_id_overlap": len(inter),
                "family_overlap": len(t) > 0 and len(e) > 0,
                "overlap_ids": sorted(inter)[:20],
            }
        )

    manifest = {
        "train_task_ids_sha256": _sha256_sorted(train_ids),
        "eval_task_ids_sha256": _sha256_sorted(eval_ids),
        "train_count": len(train_set),
        "eval_count": len(eval_set),
    }

    report = {
        "oracle_root": oracle_root,
        "eval_batches": eval_batches,
        "exclude_benchmarks": list(skip_benches),
        "dpo_pairs": len(pairs),
        "exact_task_overlap": len(overlap),
        "overlap_task_ids": sorted(overlap),
        "per_benchmark": per_bench,
        "manifest": manifest,
        "verdict": "PASS" if len(overlap) == 0 else "FAIL",
        "family_overlap_benchmarks": [r["benchmark"] for r in per_bench if r["family_overlap"]],
        "mitigation_if_fail": (
            "Retrain DPO excluding overlapping task IDs or hold out eval benchmark families"
        ),
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
