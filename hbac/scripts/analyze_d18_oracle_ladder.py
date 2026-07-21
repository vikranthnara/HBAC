"""Oracle eval ladder for D18 starvation-penalty L1 checkpoints (no inference guardrail)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.heuristics import TypePriorAllocator
from hbac.scripts.eval_compose_v2 import _bootstrap_ci
from hbac.training.batch_curriculum import load_batches
from hbac.training.l1_batch_reward import l1_schema_reward, starvation_rate
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import _resolve_l2

app = typer.Typer(help="D18 oracle ladder: hbac_d18 without fairness_reserve_alloc")


def _eval_l1(
    name: str,
    l1: Level1Policy,
    batches,
    l2,
    oracle_index: OracleIndex,
    *,
    post_guardrail=None,
) -> dict:
    successes: list[bool] = []
    starve_rates: list[float] = []
    rewards: list[float] = []
    for batch in batches:
        alloc = l1.allocate_schema(batch, int(np.argmax(l1.schema_probs(batch))))
        if post_guardrail:
            alloc = post_guardrail(alloc, batch)
        starve_rates.append(starvation_rate(alloc, batch, hard_min_frac=0.15))
        results = [
            rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        successes.extend(r.success for r in results)
        rewards.append(l1_schema_reward(results, batch, alloc))
    n = max(len(successes), 1)
    return {
        "name": name,
        "pass_at_1": sum(successes) / n,
        "pass_at_1_ci95": list(_bootstrap_ci(successes)),
        "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_starvation_rate": float(np.mean(starve_rates)) if starve_rates else 0.0,
        "num_tasks": len(successes),
        "inference_guardrail": post_guardrail is not None,
    }


@app.command()
def main(
    batches_path: str = typer.Option("checkpoints/eval_real/batches.jsonl"),
    l2_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"
    ),
    oracle_path: str = typer.Option("data/oracles/real_eval/latest"),
    baseline_l1: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
    ),
    d18_l1: str = typer.Option(
        "checkpoints/phase3_fairness_0.5/20260706T220026Z/stage3/level1_policy.npz"
    ),
    output: str = typer.Option("results/d18_oracle_ladder.json"),
) -> None:
    from hbac.training.scarcity import fairness_reserve_alloc

    batches = load_batches(Path(batches_path))
    l2 = _resolve_l2(Path(l2_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    l1_base = Level1Policy.load(Path(baseline_l1))
    l1_d18 = Level1Policy.load(Path(d18_l1))

    # type_prior baseline
    tp_succ: list[bool] = []
    for batch in batches:
        alloc = TypePriorAllocator().allocate(batch.tasks, batch.global_budget)
        for task in batch.tasks:
            r = rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            tp_succ.append(r.success)
    tp_n = max(len(tp_succ), 1)
    type_prior_p = sum(tp_succ) / tp_n

    rows = [
        _eval_l1("hbac_joint", l1_base, batches, l2, oracle_index),
        _eval_l1("hbac_d18", l1_d18, batches, l2, oracle_index),
        _eval_l1(
            "hbac_d18_guardrail",
            l1_d18,
            batches,
            l2,
            oracle_index,
            post_guardrail=lambda a, b: fairness_reserve_alloc(a, b, hard_min_frac=0.15),
        ),
        _eval_l1(
            "hbac_guardrail",
            l1_base,
            batches,
            l2,
            oracle_index,
            post_guardrail=lambda a, b: fairness_reserve_alloc(a, b, hard_min_frac=0.15),
        ),
    ]

    for row in rows:
        row["gap_vs_type_prior_pp"] = (row["pass_at_1"] - type_prior_p) * 100
        row["beats_type_prior"] = row["pass_at_1"] > type_prior_p

    d18 = next(r for r in rows if r["name"] == "hbac_d18")
    report = {
        "batches_path": batches_path,
        "oracle_path": oracle_path,
        "type_prior_pass_at_1": type_prior_p,
        "rows": rows,
        "d18_starvation_rate": d18["mean_starvation_rate"],
        "d18_beats_type_prior": d18["beats_type_prior"],
        "guardrail_still_needed": d18["mean_starvation_rate"] > 0.2 and not d18["beats_type_prior"],
        "verdict": (
            "D18_LEARNED_FAIRNESS"
            if d18["mean_starvation_rate"] < 0.2
            else "GUARDRAIL_STILL_REQUIRED"
        ),
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
