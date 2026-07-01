"""Counterfactual allocation credit for Level-1 (Research Plan §5.1, H6)."""

from __future__ import annotations

from dataclasses import dataclass

from hbac.training.batch_curriculum import TrainingBatch
from hbac.training.batch_rollout import BatchRolloutResult
from hbac.training.controller import MonolithicController
from hbac.training.level1 import Level1Allocator
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.reward import TaskControllerReward


@dataclass
class CounterfactualCredit:
    task_id: str
    advantage: float
    batch_reward: float
    counterfactual_reward: float
def counterfactual_batch_reward(
    batch: TrainingBatch,
    allocations: dict[str, int],
    controller: MonolithicController,
    oracle_index: OracleIndex,
    *,
    task_id: str,
    cached_results: list | None = None,
    reward_fn: TaskControllerReward | None = None,
) -> float:
    """R_batch^{(-i)}: replace task i budget with uniform; reuse cached rollouts for other tasks."""
    reward_fn = reward_fn or TaskControllerReward()
    uniform = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
    modified = dict(allocations)
    modified[task_id] = uniform[task_id]

    results = []
    cache_by_id = {r.task_id: r for r in (cached_results or [])}
    for task in batch.tasks:
        budget = modified[task.task_id]
        if task.task_id != task_id and task.task_id in cache_by_id and cache_by_id[task.task_id].budget == budget:
            results.append(cache_by_id[task.task_id])
        else:
            results.append(
                rollout_task_with_oracle(task, budget, controller, oracle_index, reward_fn)
            )
    return l1_schema_reward(results, batch, modified)


def compute_counterfactual_credits(
    batch: TrainingBatch,
    result: BatchRolloutResult,
    controller: MonolithicController,
    oracle_index: OracleIndex,
    *,
    cached_results: list | None = None,
) -> list[CounterfactualCredit]:
    base = result.batch_reward
    if cached_results is None:
        cached_results = result.task_results
    credits: list[CounterfactualCredit] = []
    for task in batch.tasks:
        cf = counterfactual_batch_reward(
            batch,
            result.allocations,
            controller,
            oracle_index,
            task_id=task.task_id,
            cached_results=cached_results,
        )
        credits.append(
            CounterfactualCredit(
                task_id=task.task_id,
                advantage=base - cf,
                batch_reward=base,
                counterfactual_reward=cf,
            )
        )
    return credits


def credit_weighted_schema_reward(
    credits: list[CounterfactualCredit],
    base_reward: float,
    *,
    mix: float = 0.3,
) -> float:
    """Blend batch reward with mean counterfactual advantage signal."""
    if not credits:
        return base_reward
    mean_adv = sum(c.advantage for c in credits) / len(credits)
    return (1 - mix) * base_reward + mix * (base_reward + mean_adv)
