"""Diagnose uniform vs HBAC pass@1 split under LoRA (greedy, per-task breakdown)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.scripts.eval_compose_live import LIVE_MIN_PER_TASK, _filter_batches
from hbac.training.batch_curriculum import load_batches
from hbac.training.level1 import Level1Policy
from hbac.training.phase3_pipeline import _resolve_l2

app = typer.Typer(help="Per-task uniform vs HBAC success under LoRA")


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl"),
    l2_checkpoint: str = typer.Option(..., help="Frozen L2 checkpoint"),
    l1_checkpoint: str = typer.Option(..., help="HBAC L1 .npz"),
    lora_path: str = typer.Option(..., help="LoRA adapter directory"),
    output: str = typer.Option("results/uniform_lora_diagnosis.json", help="Output JSON"),
    max_batches: int = typer.Option(10, help="Batches to diagnose (cost control)"),
) -> None:
    from hbac.core.config import LLMConfig
    from hbac.core.llm import LLMBackend
    from hbac.training.level1 import Level1Allocator

    batches = _filter_batches(
        load_batches(Path(batches_path)),
        benchmarks={"mock", "swe_bench", "tau_bench", "toolbench"},
        max_batches=max_batches,
        live_min_per_task=LIVE_MIN_PER_TASK,
    )
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    cfg = LLMConfig(lora_path=lora_path)
    llm = LLMBackend.from_config(cfg)
    clear = CLEARAllocator()

    def _task_outcomes(name: str, alloc_fn) -> list[dict]:
        from hbac.training.batch_rollout import rollout_task
        from hbac.training.reward import TaskControllerReward

        reward_fn = TaskControllerReward()
        rows: list[dict] = []
        for batch in batches:
            alloc = alloc_fn(batch)
            for task in batch.tasks:
                budget = alloc[task.task_id]
                r = rollout_task(task, budget, l2, reward_fn, llm=llm)
                rows.append(
                    {
                        "task_id": task.task_id,
                        "benchmark": task.benchmark,
                        "budget": budget,
                        "success": r.success,
                        "tokens_used": r.tokens_used,
                    }
                )
        return rows

    sid = int(__import__("numpy").argmax(l1.schema_probs(batches[0])))
    uniform_rows = _task_outcomes(
        "uniform",
        lambda b: Level1Allocator(b.global_budget).allocate(b.task_ids),
    )
    hbac_rows = _task_outcomes(
        "hbac",
        lambda b: l1.allocate_schema(b, sid),
    )
    clear_rows = _task_outcomes(
        "clear",
        lambda b: clear.allocate(b.tasks, b.global_budget),
    )

    def _index(rows: list[dict]) -> dict[str, dict]:
        return {f"{r['benchmark']}:{r['task_id']}": r for r in rows}

    u_idx, h_idx, c_idx = _index(uniform_rows), _index(hbac_rows), _index(clear_rows)
    keys = sorted(set(u_idx) | set(h_idx))
    flips: list[dict] = []
    for key in keys:
        u, h = u_idx.get(key), h_idx.get(key)
        if not u or not h:
            continue
        if u["success"] != h["success"]:
            flips.append(
                {
                    "key": key,
                    "uniform_success": u["success"],
                    "hbac_success": h["success"],
                    "uniform_budget": u["budget"],
                    "hbac_budget": h["budget"],
                    "uniform_tokens": u["tokens_used"],
                    "hbac_tokens": h["tokens_used"],
                    "clear_success": c_idx.get(key, {}).get("success"),
                }
            )

    report = {
        "lora_path": lora_path,
        "num_tasks": len(keys),
        "pass_at_1": {
            "uniform": sum(1 for r in uniform_rows if r["success"]) / max(len(uniform_rows), 1),
            "hbac": sum(1 for r in hbac_rows if r["success"]) / max(len(hbac_rows), 1),
            "clear": sum(1 for r in clear_rows if r["success"]) / max(len(clear_rows), 1),
        },
        "mean_budget": {
            "uniform": sum(r["budget"] for r in uniform_rows) / max(len(uniform_rows), 1),
            "hbac": sum(r["budget"] for r in hbac_rows) / max(len(hbac_rows), 1),
        },
        "success_flips_uniform_vs_hbac": len(flips),
        "flips_sample": flips[:20],
        "interpretation": (
            "Greedy decoding (temperature=0). Pass@1 gaps come from per-task budget "
            "differences under LoRA, not allocator eval bugs."
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
