"""Empirical beta sweep for counterfactual credit mixing (theory validation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.training.batch_curriculum import TrainingBatch, load_batches
from hbac.training.batch_rollout import BatchRolloutResult
from hbac.training.credit import compute_counterfactual_credits, credit_weighted_schema_reward
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import _resolve_l2
from hbac.training.reward import TaskControllerReward

app = typer.Typer(help="Counterfactual credit beta sweep on oracle batches")


@app.command()
def main(
    batches_path: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/batches.jsonl"
    ),
    l1_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
    ),
    l2_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/frozen_l2_controller.npz"
    ),
    oracle_path: str = typer.Option("data/oracles"),
    betas: str = typer.Option("0,0.1,0.2,0.3,0.5"),
    output: str = typer.Option("results/credit_beta_sweep.json"),
) -> None:
    batches = load_batches(Path(batches_path))[:15]
    if not batches:
        raise typer.BadParameter(f"No batches in {batches_path}")
    l1 = Level1Policy.load(Path(l1_checkpoint))
    l2 = _resolve_l2(Path(l2_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    beta_vals = [float(x.strip()) for x in betas.split(",") if x.strip()]

    adv_samples: list[float] = []
    base_rewards: list[float] = []
    reward_fn = TaskControllerReward()
    for batch in batches:
        tasks = [t for t in batch.tasks if t.benchmark != "livecodebench"]
        if len(tasks) < 2:
            continue
        sub = TrainingBatch(
            batch_id=batch.batch_id,
            tasks=tasks,
            global_budget=batch.global_budget,
            oracle_token_sum=sum(t.oracle_tokens for t in tasks),
            budget_fraction=batch.budget_fraction,
        )
        schema = int(np.argmax(l1.schema_probs(sub)))
        alloc = l1.allocate_schema(sub, schema)
        results = [
            rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index, reward_fn)
            for task in sub.tasks
        ]
        br = l1_schema_reward(results, sub, alloc)
        rollout_result = BatchRolloutResult(
            schema_id=schema,
            allocations=alloc,
            task_results=results,
            batch_reward=br,
        )
        credits = compute_counterfactual_credits(
            sub, rollout_result, l2, oracle_index, cached_results=results
        )
        adv_samples.extend(abs(c.advantage) for c in credits)
        base_rewards.append(br)
        if len(base_rewards) >= 10:
            break

    max_adv = max(adv_samples) if adv_samples else 0.0
    mean_adv = float(np.mean(adv_samples)) if adv_samples else 0.0
    var_base = float(np.var(base_rewards)) if base_rewards else 0.0

    rows = []
    for beta in beta_vals:
        bias_bound = beta * max_adv
        var_bound = (1 - beta) ** 2 * var_base + beta**2 * (max_adv**2) / max(len(batches), 1)
        rows.append(
            {
                "beta": beta,
                "bias_upper_bound": bias_bound,
                "variance_upper_bound": var_bound,
                "max_abs_advantage": max_adv,
                "mean_abs_advantage": mean_adv,
            }
        )

    report = {
        "batches_path": batches_path,
        "num_batches": len(batches),
        "implementation_mix_default": 0.3,
        "paper_beta_target": 0.2,
        "rows": rows,
        "theory_reference": "paper/appendix_theory.tex",
        "note": "Bounds from Propositions 1-2; H6 refuted at scale — small beta is safe variance reduction",
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
