"""Supervised warm-start for Variant A stop head before PPO."""

from __future__ import annotations

import math

import numpy as np

from hbac.training.controller import MonolithicController, featurize_observation
from hbac.training.probes import hacking_probe_observations


def sft_warmstart_stop_head(
    controller: MonolithicController,
    examples: list[dict],
    *,
    epochs: int = 150,
    lr: float = 0.15,
    include_probe_negatives: bool = True,
) -> list[float]:
    """Train stop head (w2, b2) via logistic regression on oracle labels."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for ex in examples:
        xs.append(featurize_observation(ex["observation"]))
        ys.append(1.0 if ex["stop"] else 0.0)

    if include_probe_negatives:
        for obs in hacking_probe_observations():
            for _ in range(3):  # up-weight anti-hacking negatives
                xs.append(featurize_observation(obs))
                ys.append(0.0)

    losses: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        grad_w2 = np.zeros_like(controller.w2)
        grad_b2 = 0.0

        for x, y in zip(xs, ys, strict=True):
            h = np.tanh(x @ controller.w1 + controller.b1)
            logit = float(h @ controller.w2 + controller.b2)
            p = 1.0 / (1.0 + math.exp(-logit))
            p = min(max(p, 1e-8), 1 - 1e-8)
            total_loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
            d_logit = p - y
            grad_w2 += h * d_logit
            grad_b2 += d_logit

        n = max(len(xs), 1)
        controller.w2 -= lr * grad_w2 / n
        controller.b2 -= lr * grad_b2 / n
        losses.append(total_loss / n)

    return losses


def init_continue_bias(controller: MonolithicController, bias: float = -4.0) -> None:
    """Strong prior against stopping before training (anti hacking)."""
    controller.b2 = bias
    controller.w2 *= 0.01
