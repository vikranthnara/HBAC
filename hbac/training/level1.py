"""Level-1 batch budget allocator and learnable policy (Phase 3)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hbac.training.batch_curriculum import BatchTask, TrainingBatch

SCHEMA_NAMES = ("uniform", "code_heavy", "tool_heavy", "shadow_price")


@dataclass
class Level1Allocator:
    """Equal-split batch allocator with optional difficulty weighting."""

    global_budget: int

    def allocate(self, task_ids: list[str], *, difficulty: dict[str, float] | None = None) -> dict[str, int]:
        if not task_ids:
            return {}
        n = len(task_ids)
        base = self.global_budget // n
        remainder = self.global_budget % n
        out: dict[str, int] = {}
        for i, tid in enumerate(task_ids):
            weight = 1.0
            if difficulty and tid in difficulty:
                weight = max(0.5, min(2.0, difficulty[tid]))
            out[tid] = int(base * weight) + (1 if i < remainder else 0)
        total = sum(out.values())
        if total > self.global_budget:
            scale = self.global_budget / total
            out = {k: max(1, int(v * scale)) for k, v in out.items()}
        return out

    def batch_violation_rate(self, allocations: dict[str, int], actual_tokens: dict[str, int]) -> float:
        if not allocations:
            return 0.0
        batch_budget = sum(allocations.values())
        batch_used = sum(actual_tokens.values())
        per_task_violations = sum(
            1 for tid, alloc in allocations.items() if actual_tokens.get(tid, 0) > alloc
        )
        batch_violated = int(batch_used > batch_budget)
        return (per_task_violations + batch_violated) / (len(allocations) + 1)


def _project_simplex(weights: np.ndarray, total: int, min_per_task: int = 1) -> dict[int, int]:
    """Map softmax weights to integer budgets summing to <= total."""
    n = len(weights)
    if n == 0:
        return {}
    w = np.maximum(weights, 1e-8)
    w = w / w.sum()
    raw = w * total
    floors = np.maximum(np.floor(raw).astype(int), min_per_task)
    alloc = floors.copy()
    remainder = total - int(alloc.sum())
    frac = raw - floors
    order = np.argsort(-frac)
    i = 0
    while remainder > 0 and n > 0:
        alloc[order[i % n]] += 1
        remainder -= 1
        i += 1
    return {int(j): int(alloc[j]) for j in range(n)}


def featurize_batch(batch: TrainingBatch) -> np.ndarray:
    """Batch-level features for Level-1 policy."""
    domains = {"swe_bench": 0, "livecodebench": 1, "toolbench": 2, "tau_bench": 3}
    counts = [0.0, 0.0, 0.0, 0.0]
    mean_diff = 0.0
    for t in batch.tasks:
        counts[domains.get(t.benchmark, 0)] += 1.0
        mean_diff += t.difficulty
    n = max(len(batch.tasks), 1)
    mean_diff /= n
    return np.array(
        [
            1.0,
            n / 20.0,
            batch.global_budget / 100_000.0,
            batch.budget_fraction,
            mean_diff,
            *counts,
        ],
        dtype=np.float64,
    )


class Level1Policy:
    """
    Learnable Level-1 allocator: softmax over G schema logits, each schema
    maps to a budget allocation template.
    """

    def __init__(self, input_dim: int = 9, num_schemas: int = 4, hidden_dim: int = 32) -> None:
        self.input_dim = input_dim
        self.num_schemas = num_schemas
        self.hidden_dim = hidden_dim
        scale = 0.05
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, num_schemas) * scale
        # Bias away from uniform schema (0) to defeat mode-collapse at init
        self.b2 = np.array([0.0, 0.4, 0.4, 0.2], dtype=np.float64)[:num_schemas]
        if len(self.b2) < num_schemas:
            self.b2 = np.pad(self.b2, (0, num_schemas - len(self.b2)))

    def schema_logits(self, batch: TrainingBatch) -> np.ndarray:
        x = featurize_batch(batch)
        h = np.tanh(x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def schema_probs(self, batch: TrainingBatch) -> np.ndarray:
        logits = self.schema_logits(batch)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / exp.sum()

    def log_prob_schema(self, batch: TrainingBatch, schema_id: int) -> float:
        p = self.schema_probs(batch)
        p = min(max(float(p[schema_id]), 1e-8), 1 - 1e-8)
        return math.log(p)

    def allocate_schema(self, batch: TrainingBatch, schema_id: int) -> dict[str, int]:
        tasks = batch.tasks
        n = len(tasks)
        if n == 0:
            return {}
        total = batch.global_budget
        tid_list = [t.task_id for t in tasks]
        difficulty = {t.task_id: t.difficulty for t in tasks}

        if schema_id % self.num_schemas == 0:
            return Level1Allocator(total).allocate(tid_list, difficulty=difficulty)

        weights = np.ones(n, dtype=np.float64)
        for i, t in enumerate(tasks):
            if schema_id % self.num_schemas == 1:
                if t.benchmark in {"livecodebench", "swe_bench"}:
                    weights[i] = 2.5
                else:
                    weights[i] = 0.5
            elif schema_id % self.num_schemas == 2:
                if t.benchmark in {"toolbench", "tau_bench"}:
                    weights[i] = 2.5
                else:
                    weights[i] = 0.5
            else:
                weights[i] = t.difficulty

        idx_map = _project_simplex(weights, total)
        return {tid_list[i]: idx_map[i] for i in range(n)}

    def sample_schemas(self, batch: TrainingBatch, num_groups: int) -> list[int]:
        """Stratified GRPO groups: always include non-uniform schemas for reward variance."""
        rng = np.random.default_rng()
        core = [i % self.num_schemas for i in range(1, min(self.num_schemas, 4))]
        out = list(core)
        probs = self.schema_probs(batch)
        while len(out) < num_groups:
            out.append(int(rng.choice(self.num_schemas, p=probs)))
        return out[:num_groups]

    def flat_params(self) -> np.ndarray:
        return np.concatenate(
            [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), self.b2.ravel()]
        )

    def load_flat_params(self, params: np.ndarray) -> None:
        idx = 0
        w1_size = self.input_dim * self.hidden_dim
        self.w1 = params[idx : idx + w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        self.b1 = params[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        w2_size = self.hidden_dim * self.num_schemas
        self.w2 = params[idx : idx + w2_size].reshape(self.hidden_dim, self.num_schemas)
        idx += w2_size
        self.b2 = params[idx : idx + self.num_schemas]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            input_dim=self.input_dim,
            num_schemas=self.num_schemas,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def load(cls, path: Path) -> Level1Policy:
        data = np.load(path)
        pol = cls(int(data["input_dim"]), int(data["num_schemas"]), int(data["hidden_dim"]))
        pol.w1 = data["w1"]
        pol.b1 = data["b1"]
        pol.w2 = data["w2"]
        pol.b2 = data["b2"]
        return pol

    def frozen_copy(self) -> Level1Policy:
        other = Level1Policy(self.input_dim, self.num_schemas, self.hidden_dim)
        other.w1 = np.array(self.w1, copy=True)
        other.b1 = np.array(self.b1, copy=True)
        other.w2 = np.array(self.w2, copy=True)
        other.b2 = np.array(self.b2, copy=True)
        return other
