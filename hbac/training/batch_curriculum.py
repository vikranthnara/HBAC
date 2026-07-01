"""Heterogeneous batched workloads with budget curriculum (Phase 3)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hbac.core.trajectory import TrajectoryStore
from hbac.training.dataset import find_oracle_paths

BENCHMARK_MIX = {
    "swe_bench": 2,
    "livecodebench": 4,
    "toolbench": 2,
    "tau_bench": 2,
}

BUDGET_FRACTIONS = (0.70, 0.55, 0.40)  # tight curriculum for Pass@1 differentiation


@dataclass
class BatchTask:
    task_id: str
    benchmark: str
    oracle_tokens: int
    difficulty: float = 1.0


@dataclass
class TrainingBatch:
    batch_id: str
    tasks: list[BatchTask]
    global_budget: int
    oracle_token_sum: int
    budget_fraction: float

    @property
    def task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks]

    def to_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "global_budget": self.global_budget,
            "oracle_token_sum": self.oracle_token_sum,
            "budget_fraction": self.budget_fraction,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "benchmark": t.benchmark,
                    "oracle_tokens": t.oracle_tokens,
                    "difficulty": t.difficulty,
                }
                for t in self.tasks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingBatch:
        tasks = [
            BatchTask(
                task_id=t["task_id"],
                benchmark=t["benchmark"],
                oracle_tokens=int(t["oracle_tokens"]),
                difficulty=float(t.get("difficulty", 1.0)),
            )
            for t in data["tasks"]
        ]
        return cls(
            batch_id=data["batch_id"],
            tasks=tasks,
            global_budget=int(data["global_budget"]),
            oracle_token_sum=int(data["oracle_token_sum"]),
            budget_fraction=float(data["budget_fraction"]),
        )


def _load_oracle_tasks(oracle_root: Path) -> dict[str, list[BatchTask]]:
    by_benchmark: dict[str, list[BatchTask]] = {b: [] for b in BENCHMARK_MIX}
    seen: set[tuple[str, str]] = set()

    for path in find_oracle_paths(oracle_root):
        for traj in TrajectoryStore(path).load_successful():
            key = (traj.benchmark, traj.task_id)
            if key in seen:
                continue
            seen.add(key)
            bench = traj.benchmark
            if bench not in by_benchmark:
                by_benchmark[bench] = []
            diff = min(2.0, max(0.5, traj.total_tokens / 500.0))
            by_benchmark[bench].append(
                BatchTask(
                    task_id=traj.task_id,
                    benchmark=bench,
                    oracle_tokens=traj.total_tokens,
                    difficulty=diff,
                )
            )
    return by_benchmark


def _fallback_tasks() -> dict[str, list[BatchTask]]:
    """Deterministic fallbacks when oracle pool is thin."""
    from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES

    out: dict[str, list[BatchTask]] = {b: [] for b in BENCHMARK_MIX}
    for ep in DETERMINISTIC_EPISODES:
        out.setdefault(ep.env_name, []).append(
            BatchTask(
                task_id=ep.task_id,
                benchmark=ep.env_name,
                oracle_tokens=200,
                difficulty=1.0,
            )
        )
    return out


def sample_batch(
    oracle_root: Path,
    *,
    budget_fraction: float = 0.90,
    mix: dict[str, int] | None = None,
    seed: int | None = None,
) -> TrainingBatch:
    mix = mix or BENCHMARK_MIX
    rng = random.Random(seed)
    pool = _load_oracle_tasks(oracle_root)
    fallback = _fallback_tasks()

    selected: list[BatchTask] = []
    for benchmark, count in mix.items():
        candidates = list(pool.get(benchmark, []))
        # Pad with deterministic stubs so every batch is domain-heterogeneous
        fb = fallback.get(benchmark, [])
        for t in fb:
            if t.task_id not in {c.task_id for c in candidates}:
                candidates.append(t)
        if not candidates:
            candidates = fb
        if not candidates:
            continue
        if len(candidates) >= count:
            selected.extend(rng.sample(candidates, count))
        else:
            selected.extend(rng.choices(candidates, k=count))

    oracle_sum = sum(t.oracle_tokens for t in selected) or 1
    min_budget = len(selected) * 40
    global_budget = max(min_budget, int(oracle_sum * budget_fraction))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_id = f"batch-{run_id}-{seed or 0}"

    return TrainingBatch(
        batch_id=batch_id,
        tasks=selected,
        global_budget=global_budget,
        oracle_token_sum=oracle_sum,
        budget_fraction=budget_fraction,
    )


def generate_curriculum_batches(
    oracle_root: Path,
    *,
    num_batches: int = 30,
    seed: int = 42,
) -> list[TrainingBatch]:
    rng = random.Random(seed)
    batches: list[TrainingBatch] = []
    for i in range(num_batches):
        frac = BUDGET_FRACTIONS[i % len(BUDGET_FRACTIONS)]
        batches.append(
            sample_batch(oracle_root, budget_fraction=frac, seed=rng.randint(0, 10_000_000))
        )
    return batches


def save_batches(batches: list[TrainingBatch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for b in batches:
            f.write(json.dumps(b.to_dict()) + "\n")


def load_batches(path: Path) -> list[TrainingBatch]:
    batches: list[TrainingBatch] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batches.append(TrainingBatch.from_dict(json.loads(line)))
    return batches
