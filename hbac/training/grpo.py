"""GRPO trainer for prototype Level-1/Level-2 policies (Phase 3, Variant B)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hbac.training.level1 import Level1Policy


@dataclass
class GRPOStats:
    mean_reward: float
    std_reward: float
    policy_loss: float
    skipped: bool
    entropy: float


def group_advantages(rewards: list[float], eps: float = 1e-6) -> tuple[np.ndarray, bool]:
    """GRPO relative advantages [A17]: A_g = (R_g - mean) / (std + eps)."""
    if not rewards:
        return np.array([]), True
    arr = np.array(rewards, dtype=np.float64)
    std = float(arr.std())
    if std < eps:
        return np.zeros_like(arr), True
    mean = float(arr.mean())
    return (arr - mean) / (std + eps), False


class GRPOTrainer:
    """Policy-gradient update with group-relative advantages and optional KL to ref."""

    def __init__(
        self,
        policy: Level1Policy,
        *,
        learning_rate: float = 0.01,
        kl_coef: float = 0.02,
        entropy_coef: float = 0.05,
    ) -> None:
        self.policy = policy
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.ref_policy = policy.frozen_copy()

    def _entropy(self, batch) -> float:
        p = self.policy.schema_probs(batch)
        return float(-np.sum(p * np.log(p + 1e-8)))

    def _grad_log_prob_schema(self, batch, schema_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from hbac.training.level1 import featurize_batch

        x = featurize_batch(batch)
        h = np.tanh(x @ self.policy.w1 + self.policy.b1)
        logits = h @ self.policy.w2 + self.policy.b2
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / exp.sum()
        dlogits = -probs
        dlogits[schema_id] += 1.0
        grad_w2 = np.outer(h, dlogits)
        grad_b2 = dlogits.copy()
        dh = dlogits @ self.policy.w2.T
        dtanh = dh * (1.0 - h**2)
        grad_w1 = np.outer(x, dtanh)
        grad_b1 = dtanh.copy()
        return grad_w1, grad_b1, grad_w2, grad_b2

    def update_l1(
        self,
        batch,
        schema_ids: list[int],
        rewards: list[float],
    ) -> GRPOStats:
        advantages, skipped = group_advantages(rewards)
        mean_r = float(np.mean(rewards)) if rewards else 0.0
        std_r = float(np.std(rewards)) if rewards else 0.0

        if skipped:
            return GRPOStats(mean_r, std_r, 0.0, True, self._entropy(batch))

        grad_w1 = np.zeros_like(self.policy.w1)
        grad_b1 = np.zeros_like(self.policy.b1)
        grad_w2 = np.zeros_like(self.policy.w2)
        grad_b2 = np.zeros_like(self.policy.b2)

        for schema_id, adv in zip(schema_ids, advantages, strict=True):
            gw1, gb1, gw2, gb2 = self._grad_log_prob_schema(batch, schema_id)
            grad_w1 += adv * gw1
            grad_b1 += adv * gb1
            grad_w2 += adv * gw2
            grad_b2 += adv * gb2

        ent = self._entropy(batch)
        params = self.policy.flat_params()

        flat_grad = np.concatenate(
            [
                grad_w1.ravel(),
                grad_b1.ravel(),
                grad_w2.ravel(),
                grad_b2.ravel(),
            ]
        )
        flat_grad /= max(len(schema_ids), 1)

        params = params + self.learning_rate * flat_grad
        self.policy.load_flat_params(params)

        loss = -float(np.mean(advantages))
        return GRPOStats(mean_r, std_r, loss, False, ent)

    def _entropy_grad(self, batch, params: np.ndarray, delta: float) -> np.ndarray:
        base = self._entropy(batch)
        grad = np.zeros_like(params)
        for j in range(len(params)):
            p_plus = params.copy()
            p_plus[j] += delta
            self.policy.load_flat_params(p_plus)
            grad[j] = (self._entropy(batch) - base) / delta
        self.policy.load_flat_params(params)
        return grad


class L2GRPOTrainer:
    """GRPO updates for Level-2 stop head from grouped stop-label examples."""

    def __init__(
        self,
        controller,
        *,
        learning_rate: float = 0.01,
        num_samples: int = 8,
        eps: float = 1e-6,
    ) -> None:
        from hbac.training.controller import MonolithicController

        self.controller: MonolithicController = controller
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.eps = eps
        self.ref = controller.frozen_copy()

    def update_from_examples(self, examples: list[dict]) -> GRPOStats:
        """Sample G threshold perturbations; GRPO on mean per-example reward."""
        if not examples:
            return GRPOStats(0.0, 0.0, 0.0, True, 0.0)

        rewards: list[float] = []
        base_params = self.controller.flat_params().copy()

        for g in range(self.num_samples):
            noise = np.random.randn(*base_params.shape) * 0.002 * (g + 1)
            self.controller.load_flat_params(base_params + noise)
            correct = 0
            for ex in examples:
                pred = self.controller.should_stop(ex["observation"])
                if pred == ex["stop"]:
                    correct += 1
            rewards.append(correct / len(examples))

        advantages, skipped = group_advantages(rewards, self.eps)
        if skipped:
            self.controller.load_flat_params(base_params)
            return GRPOStats(float(np.mean(rewards)), 0.0, 0.0, True, 0.0)

        params = base_params.copy()
        grad = np.zeros_like(params)
        delta = 1e-5
        mean_adv = float(np.mean(advantages))

        for j in range(len(params)):
            p_plus = params.copy()
            p_plus[j] += delta
            self.controller.load_flat_params(p_plus)
            r_plus = sum(
                int(self.controller.should_stop(ex["observation"]) == ex["stop"]) for ex in examples
            ) / len(examples)
            self.controller.load_flat_params(params)
            grad[j] = mean_adv * (r_plus - rewards[0]) / delta

        params = params + self.learning_rate * grad
        self.controller.load_flat_params(params)
        return GRPOStats(
            float(np.mean(rewards)),
            float(np.std(rewards)),
            -mean_adv,
            False,
            0.0,
        )
