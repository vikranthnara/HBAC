"""Apply live-style per-task floors to training batches (P3: oracle/live alignment)."""

from __future__ import annotations

from hbac.training.batch_curriculum import TrainingBatch, load_batches


def apply_live_floor(
    batches: list[TrainingBatch],
    *,
    live_min_per_task: int = 600,
    budget_fraction: float | None = None,
    live: bool = True,
) -> list[TrainingBatch]:
    """Mirror eval_compose_live._filter_batches budget logic without benchmark filter."""
    out: list[TrainingBatch] = []
    for batch in batches:
        tasks = list(batch.tasks)
        if not tasks:
            continue
        oracle_sum = sum(t.oracle_tokens for t in tasks) or 1
        n = len(tasks)
        frac = budget_fraction if budget_fraction is not None else batch.budget_fraction
        frac_budget = int(oracle_sum * frac)
        full_oracle = batch.oracle_token_sum or 1
        scaled_train_budget = int(batch.global_budget * (oracle_sum / full_oracle))
        if budget_fraction is not None and full_oracle:
            scaled_train_budget = int(oracle_sum * frac)
        floor = n * (live_min_per_task if live else 40)
        out.append(
            TrainingBatch(
                batch_id=batch.batch_id,
                tasks=tasks,
                global_budget=max(floor, scaled_train_budget, frac_budget),
                oracle_token_sum=oracle_sum,
                budget_fraction=frac,
            )
        )
    return out


def load_batches_with_floor(
    path,
    *,
    live_min_per_task: int,
    budget_fraction: float = 0.4,
) -> list[TrainingBatch]:
    return apply_live_floor(
        load_batches(path),
        live_min_per_task=live_min_per_task,
        budget_fraction=budget_fraction,
    )
