"""Comprehensive oracle compose eval: HBAC + all baselines + raw metrics (v2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.baselines.clear_official import CLEAROfficialAllocator
from hbac.baselines.heuristics import (
    BatchReFORCProxyAllocator,
    BatchTABProxyAllocator,
    DifficultyInverseAllocator,
    SJFAllocator,
    TypePriorAllocator,
)
from hbac.baselines.reforc_official import ReFORCOfficialAllocator
from hbac.baselines.zebra import ZEBRAAllocator
from hbac.baselines.zebra_official import ZEBRAOfficialAllocator
from hbac.scripts.eval_compose import _eval_clear, _eval_uniform
from hbac.eval.batch_floor import load_batches_with_floor
from hbac.training.batch_curriculum import load_batches
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy
from hbac.training.scarcity import fairness_reserve_alloc

app = typer.Typer(help="V2 baseline matrix on oracle replay (raw pass@1, tokens, violations)")


def _bootstrap_ci(successes: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not successes:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(successes, dtype=float)
    samples = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl"),
    l2_checkpoint: str = typer.Option("checkpoints/variant_a/latest"),
    l1_checkpoint: str = typer.Option(..., help="HBAC L1 .npz"),
    oracle_path: str = typer.Option("data/oracles"),
    output: str = typer.Option("results/v2_baseline_matrix_oracle.json"),
    min_per_task: int = typer.Option(40, help="Per-task floor for proxy allocators"),
    live_min_per_task: int | None = typer.Option(
        None,
        help="If set, apply live-style batch floors via load_batches_with_floor",
    ),
    budget_fraction: float = typer.Option(0.4, help="Global budget fraction of oracle sum"),
) -> None:
    if live_min_per_task is not None:
        batches = load_batches_with_floor(
            Path(batches_path),
            live_min_per_task=live_min_per_task,
            budget_fraction=budget_fraction,
        )
    else:
        batches = load_batches(Path(batches_path))
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    oracle_index = OracleIndex(Path(oracle_path))
    clear = CLEARAllocator(min_per_task=min_per_task)
    zebra = ZEBRAAllocator(min_per_task=min_per_task)

    clear_official = CLEAROfficialAllocator(min_per_task=min_per_task)
    zebra_official = ZEBRAOfficialAllocator(min_per_task=min_per_task)
    reforc_official = ReFORCOfficialAllocator(min_per_task=min_per_task)

    def _hbac_fair(batch):
        alloc = l1.allocate_schema(batch, int(np.argmax(l1.schema_probs(batch))))
        return fairness_reserve_alloc(alloc, batch, hard_min_frac=0.15)

    allocators: dict[str, object] = {
        "uniform": lambda b: Level1Allocator(b.global_budget).allocate(b.task_ids),
        "hbac": lambda b: l1.allocate_schema(b, int(np.argmax(l1.schema_probs(b)))),
        "hbac_fair": _hbac_fair,
        "clear": lambda b: clear.allocate(b.tasks, b.global_budget),
        "clear_official": lambda b: clear_official.allocate(b.tasks, b.global_budget),
        "zebra": lambda b: zebra.allocate(b.tasks, b.global_budget),
        "zebra_official": lambda b: zebra_official.allocate(b.tasks, b.global_budget),
        "reforc_official": lambda b: reforc_official.allocate(b.tasks, b.global_budget),
        "sjf": lambda b: SJFAllocator(min_per_task=1).allocate(b.tasks, b.global_budget),
        "type_prior": lambda b: TypePriorAllocator().allocate(b.tasks, b.global_budget),
        "difficulty_inverse": lambda b: DifficultyInverseAllocator(min_per_task=1).allocate(
            b.tasks, b.global_budget
        ),
        "tab_proxy": lambda b: BatchTABProxyAllocator(min_per_task=min_per_task).allocate(
            b.tasks, b.global_budget
        ),
        "reforc_proxy": lambda b: BatchReFORCProxyAllocator(min_per_task=min_per_task).allocate(
            b.tasks, b.global_budget
        ),
    }

    report: dict = {
        "allocators": {},
        "min_per_task": min_per_task,
        "num_batches": len(batches),
        "budget_fraction": budget_fraction,
        "live_min_per_task": live_min_per_task,
    }

    if "hbac" in allocators:
        m = evaluate_l1_policy(l1, batches, l2, oracle_index)
        report["allocators"]["hbac"] = {
            **m.to_dict(),
            "metric_note": "raw oracle replay; no compliant utility",
        }
        # HBAC + fairness reserve (D17)
        rewards_f: list[float] = []
        successes_f: list[bool] = []
        for batch in batches:
            alloc = _hbac_fair(batch)
            results = [
                rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
                for task in batch.tasks
            ]
            successes_f.extend(r.success for r in results)
            rewards_f.append(l1_schema_reward(results, batch, alloc))
        nf = max(len(successes_f), 1)
        lo_f, hi_f = _bootstrap_ci(successes_f)
        report["allocators"]["hbac_fair"] = {
            "pass_at_1": sum(successes_f) / nf,
            "pass_at_1_ci95": [lo_f, hi_f],
            "mean_batch_reward": float(np.mean(rewards_f)) if rewards_f else 0.0,
            "num_tasks": len(successes_f),
            "note": "L1 + fairness_reserve_alloc(hard_min_frac=0.15)",
        }

    for name in ("uniform", "clear"):
        if name == "clear":
            report["allocators"]["clear"] = _eval_clear(batches, l2, oracle_index, clear)
        else:
            report["allocators"]["uniform"] = _eval_uniform(batches, l2, oracle_index)

    for name, alloc_fn in allocators.items():
        if name in ("hbac", "hbac_fair", "uniform", "clear"):
            continue
        rewards: list[float] = []
        successes: list[bool] = []
        violations = 0
        tokens: list[int] = []
        for batch in batches:
            alloc = alloc_fn(batch)
            results = [
                rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
                for task in batch.tasks
            ]
            successes.extend(r.success for r in results)
            tokens.extend(r.tokens_used for r in results)
            rewards.append(l1_schema_reward(results, batch, alloc))
            violations += sum(1 for r in results if r.budget_violated)
        n = max(len(successes), 1)
        lo, hi = _bootstrap_ci(successes)
        report["allocators"][name] = {
            "pass_at_1": sum(successes) / n,
            "pass_at_1_ci95": [lo, hi],
            "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
            "batch_violation_rate": violations / (n + len(batches)),
            "mean_tokens_used": float(np.mean(tokens)) if tokens else 0.0,
            "num_tasks": len(successes),
        }

    # HBAC vs best heuristic on pass@1
    hb_p = report["allocators"]["hbac"]["pass_at_1"]
    type_prior_p = report["allocators"]["type_prior"]["pass_at_1"]
    best_heur = max(
        report["allocators"][k]["pass_at_1"]
        for k in ("sjf", "type_prior", "difficulty_inverse", "tab_proxy", "reforc_proxy")
    )
    report["hbac_beats_best_heuristic_pp"] = (hb_p - best_heur) * 100
    report["hbac_ties_type_prior"] = abs(hb_p - type_prior_p) < 1e-6
    report["type_prior_higher_reward"] = (
        report["allocators"]["type_prior"]["mean_batch_reward"]
        > report["allocators"]["hbac"]["mean_batch_reward"]
    )
    report["hbac_fair_beats_type_prior"] = (
        report["allocators"].get("hbac_fair", {}).get("pass_at_1", 0) > type_prior_p
    )
    report["proxy_disclaimer"] = (
        "clear_official/zebra_official/reforc_official implement paper algorithms (Tier-A). "
        "Tier-B proxies retained for regression. Author GitHub repos not yet published."
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
