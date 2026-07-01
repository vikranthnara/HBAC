"""Utility prediction network V(q_i, b) for Variant A Level-1 (Research Plan §5.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hbac.training.batch_curriculum import BatchTask


def featurize_task(task: BatchTask, budget: int, global_budget: int) -> np.ndarray:
    domains = {"swe_bench": 0, "livecodebench": 1, "toolbench": 2, "tau_bench": 3}
    d = [0.0, 0.0, 0.0, 0.0]
    d[domains.get(task.benchmark, 0)] = 1.0
    return np.array(
        [
            1.0,
            budget / max(global_budget, 1),
            task.oracle_tokens / 10_000.0,
            task.difficulty,
            *d,
        ],
        dtype=np.float64,
    )


class UtilityNetwork:
    """Predicts expected task utility given allocation."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        scale = 0.05
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim) * scale
        self.b2 = 0.0

    def predict(self, task: BatchTask, budget: int, global_budget: int) -> float:
        x = featurize_task(task, budget, global_budget)
        h = np.tanh(x @ self.w1 + self.b1)
        return float(h @ self.w2 + self.b2)

    def allocate_greedy(
        self,
        tasks: list[BatchTask],
        global_budget: int,
        *,
        min_per_task: int = 50,
    ) -> dict[str, int]:
        """Shadow-price style: iteratively assign budget to highest marginal utility."""
        if not tasks:
            return {}
        n = len(tasks)
        remaining = global_budget
        alloc = {t.task_id: min_per_task for t in tasks}
        remaining -= min_per_task * n
        if remaining < 0:
            base = max(1, global_budget // n)
            return {t.task_id: base for t in tasks}

        candidates = list(range(n))
        while remaining > 0 and candidates:
            best_i, best_gain = candidates[0], -1e9
            for i in candidates:
                tid = tasks[i].task_id
                u1 = self.predict(tasks[i], alloc[tid] + 100, global_budget)
                u0 = self.predict(tasks[i], alloc[tid], global_budget)
                gain = u1 - u0
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            step = min(100, remaining)
            alloc[tasks[best_i].task_id] += step
            remaining -= step
        return alloc

    def flat_params(self) -> np.ndarray:
        return np.concatenate([self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), [self.b2]])

    def load_flat_params(self, params: np.ndarray) -> None:
        idx = 0
        w1_size = self.input_dim * self.hidden_dim
        self.w1 = params[idx : idx + w1_size].reshape(self.input_dim, self.hidden_dim)
        idx += w1_size
        self.b1 = params[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.w2 = params[idx : idx + self.hidden_dim]
        idx += self.hidden_dim
        self.b2 = float(params[idx])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=np.array([self.b2]),
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
        )

    @classmethod
    def load(cls, path: Path) -> UtilityNetwork:
        data = np.load(path)
        net = cls(int(data["input_dim"]), int(data["hidden_dim"]))
        net.w1 = data["w1"]
        net.b1 = data["b1"]
        net.w2 = data["w2"]
        net.b2 = float(data["b2"])
        return net
