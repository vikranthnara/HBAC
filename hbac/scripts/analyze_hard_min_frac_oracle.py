"""Oracle sweep: hbac_fair vs type_prior across hard_min_frac values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.scripts.eval_compose_v2 import _bootstrap_ci
from hbac.training.batch_curriculum import load_batches
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import _resolve_l2
from hbac.training.scarcity import fairness_reserve_alloc

app = typer.Typer(help="Oracle hard_min_frac sweep for hbac_fair")


@app.command()
def main(
    batches_path: str = typer.Option("checkpoints/eval_real/batches.jsonl"),
    l2_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"
    ),
    l1_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
    ),
    oracle_path: str = typer.Option("data/oracles/real_eval/latest"),
    fracs: str = typer.Option("0.10,0.15,0.20,0.25", help="Comma-separated hard_min_frac values"),
    output: str = typer.Option("results/hard_min_frac_oracle_sweep.json"),
) -> None:
    batches = load_batches(Path(batches_path))
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    values = [float(x.strip()) for x in fracs.split(",") if x.strip()]

    # type_prior baseline (once)
    tp_successes: list[bool] = []
    for batch in batches:
        from hbac.baselines.heuristics import TypePriorAllocator

        alloc = TypePriorAllocator().allocate(batch.tasks, batch.global_budget)
        for task in batch.tasks:
            r = rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            tp_successes.append(r.success)
    tp_n = max(len(tp_successes), 1)
    type_prior_p = sum(tp_successes) / tp_n
    tp_ci = _bootstrap_ci(tp_successes)

    rows: list[dict] = []
    for frac in values:
        successes: list[bool] = []
        rewards: list[float] = []
        for batch in batches:
            alloc = l1.allocate_schema(batch, int(np.argmax(l1.schema_probs(batch))))
            alloc = fairness_reserve_alloc(alloc, batch, hard_min_frac=frac)
            results = [
                rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
                for task in batch.tasks
            ]
            successes.extend(r.success for r in results)
            rewards.append(l1_schema_reward(results, batch, alloc))
        n = max(len(successes), 1)
        fair_p = sum(successes) / n
        lo, hi = _bootstrap_ci(successes)
        rows.append(
            {
                "hard_min_frac": frac,
                "hbac_fair_pass_at_1": fair_p,
                "hbac_fair_ci95": [lo, hi],
                "gap_vs_type_prior_pp": (fair_p - type_prior_p) * 100,
                "beats_type_prior": fair_p > type_prior_p,
                "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
                "num_tasks": len(successes),
            }
        )

    best = max(rows, key=lambda r: r["hbac_fair_pass_at_1"])
    report = {
        "batches_path": batches_path,
        "oracle_path": oracle_path,
        "type_prior_pass_at_1": type_prior_p,
        "type_prior_ci95": list(tp_ci),
        "rows": rows,
        "best_frac": best["hard_min_frac"],
        "best_pass_at_1": best["hbac_fair_pass_at_1"],
        "verdict": (
            "BEATS_TYPE_PRIOR"
            if any(r["beats_type_prior"] for r in rows)
            else "NEVER_BEATS_TYPE_PRIOR"
        ),
        "interpretation": (
            "Oracle hbac_fair does not exceed type_prior at any hard_min_frac tested; "
            "live win (+1 pp) is a generation/regime effect, not oracle allocation gap."
            if not any(r["beats_type_prior"] for r in rows)
            else "Some hard_min_frac values beat type_prior on oracle."
        ),
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
