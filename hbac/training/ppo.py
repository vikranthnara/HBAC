from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController
from hbac.training.reward import TaskControllerReward


@dataclass
class Transition:
    observation_features: np.ndarray
    stop_action: bool
    reward: float
    old_log_prob: float
    done: bool


@dataclass
class PPOStats:
    policy_loss: float
    kl_divergence: float
    entropy: float
    kl_coef: float
    ref_kl: float


class PPOTrainer:
    """
    Minimal PPO trainer for monolithic stop controller (Variant A, Stage 1).

    Uses frozen reference policy for KL penalty [A11] to prevent always-stop hacking.
    """

    def __init__(
        self,
        controller: MonolithicController,
        config: PPOConfig | None = None,
        reward_fn: TaskControllerReward | None = None,
    ) -> None:
        self.controller = controller
        self.config = config or PPOConfig()
        self.reward_fn = reward_fn or TaskControllerReward()
        self._kl_coef = self.config.kl_coef
        self.ref_controller = controller.frozen_copy()

    def _param_mask(self) -> np.ndarray:
        """Mask for trainable flat params (1=train, 0=frozen)."""
        if not self.config.freeze_hidden:
            return np.ones(self.controller.flat_params().shape[0])
        n = len(self.controller.flat_params())
        w1_size = self.controller.input_dim * self.controller.hidden_dim
        mask = np.zeros(n)
        mask[w1_size + self.controller.hidden_dim :] = 1.0  # w2 + b2 only
        return mask

    def compute_kl(self, old_log_probs: np.ndarray, new_log_probs: np.ndarray) -> float:
        """Approximate mean KL(old || new) for Bernoulli stop actions."""
        return float(np.mean(old_log_probs - new_log_probs))

    def adapt_kl_coef(self, kl: float) -> None:
        if not self.config.kl_adaptive:
            return
        if kl > 2 * self.config.kl_target:
            self._kl_coef *= 1.5
        elif kl < self.config.kl_target / 2 and self._kl_coef > 1e-4:
            self._kl_coef /= 1.5
        self._kl_coef = float(np.clip(self._kl_coef, 0.0, 1.0))

    def _forward_from_features(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        h = np.tanh(x @ self.controller.w1 + self.controller.b1)
        logit = float(h @ self.controller.w2 + self.controller.b2)
        return logit, h

    def _log_prob_from_logit(self, logit: float, stop: bool) -> float:
        p = 1.0 / (1.0 + math.exp(-logit))
        p = min(max(p, 1e-8), 1 - 1e-8)
        return math.log(p if stop else 1 - p)

    def update(self, batch: list[Transition]) -> PPOStats:
        if not batch:
            return PPOStats(0.0, 0.0, 0.0, self._kl_coef, 0.0)

        params = self.controller.flat_params()
        eps = self.config.clip_epsilon

        old_log_probs = np.array([t.old_log_prob for t in batch])
        # Imitation-style advantages from oracle labels (avoid always-stop hacking)
        advantages = np.array([1.0 if t.stop_action else -0.25 for t in batch])
        if advantages.std() > 1e-8:
            advantages = advantages / (advantages.std() + 1e-8)

        grad = np.zeros_like(params)
        new_log_probs_list = []
        ref_log_probs_list = []
        entropies = []

        for i, tr in enumerate(batch):
            logit, _ = self._forward_from_features(tr.observation_features)
            p = 1.0 / (1.0 + math.exp(-logit))
            p = min(max(p, 1e-8), 1 - 1e-8)
            new_lp = self._log_prob_from_logit(logit, tr.stop_action)
            new_log_probs_list.append(new_lp)
            ref_lp = self.ref_controller._log_prob_from_features(
                tr.observation_features, tr.stop_action
            )
            ref_log_probs_list.append(ref_lp)
            entropies.append(-(p * math.log(p) + (1 - p) * math.log(1 - p)))

            ratio = math.exp(new_lp - tr.old_log_prob)
            clipped = np.clip(ratio, 1 - eps, 1 + eps)
            coeff = advantages[i] * (1.0 if ratio <= clipped else 0.0)

            delta = 1e-5
            for j in range(len(params)):
                p_plus = params.copy()
                p_plus[j] += delta
                self.controller.load_flat_params(p_plus)
                lp_plus, _ = self._forward_from_features(tr.observation_features)
                lp_plus = self._log_prob_from_logit(lp_plus, tr.stop_action)
                grad[j] += coeff * (lp_plus - new_lp) / delta

        self.controller.load_flat_params(params)
        new_log_probs = np.array(new_log_probs_list)
        ref_log_probs = np.array(ref_log_probs_list)
        kl = self.compute_kl(old_log_probs, new_log_probs)
        ref_kl = float(np.mean(ref_log_probs - new_log_probs))
        self.adapt_kl_coef(ref_kl)

        mean_entropy = float(np.mean(entropies))
        # KL(ref || new) penalty pulls policy toward initialization
        grad -= self._kl_coef * self._ref_kl_grad(batch, new_log_probs)
        grad += self.config.entropy_coef * 0.01
        grad *= self._param_mask()

        lr = (
            self.config.learning_rate_stop_head
            if self.config.freeze_hidden
            else self.config.learning_rate
        )
        params = params + lr * grad
        norm = np.linalg.norm(params)
        if norm > 10:
            params = params * (10 / norm)
        self.controller.load_flat_params(params)

        policy_loss = -float(np.mean(advantages))
        return PPOStats(
            policy_loss=policy_loss,
            kl_divergence=kl,
            entropy=mean_entropy,
            kl_coef=self._kl_coef,
            ref_kl=ref_kl,
        )

    def _ref_kl_grad(self, batch: list[Transition], new_log_probs: np.ndarray) -> np.ndarray:
        """Gradient of mean KL(ref || new) w.r.t. controller params."""
        params = self.controller.flat_params()
        grad = np.zeros_like(params)
        delta = 1e-5
        base_kl = float(np.mean(
            [
                self.ref_controller._log_prob_from_features(t.observation_features, t.stop_action)
                for t in batch
            ]
        ) - np.mean(new_log_probs))

        for j in range(len(params)):
            p_plus = params.copy()
            p_plus[j] += delta
            self.controller.load_flat_params(p_plus)
            kl_plus = 0.0
            for tr in batch:
                logit, _ = self._forward_from_features(tr.observation_features)
                lp = self._log_prob_from_logit(logit, tr.stop_action)
                ref_lp = self.ref_controller._log_prob_from_features(
                    tr.observation_features, tr.stop_action
                )
                kl_plus += ref_lp - lp
            kl_plus /= len(batch)
            grad[j] = (kl_plus - base_kl) / delta

        self.controller.load_flat_params(params)
        return grad
