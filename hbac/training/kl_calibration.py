"""Post-SFT PPO loop tuned to land KL(ref||new) in the Phase 2 gate band."""

from __future__ import annotations

import numpy as np

from hbac.training.controller import MonolithicController
from hbac.training.ppo import PPOStats, PPOTrainer
from hbac.training.probes import early_stop_tool_rate, probe_premature_stop_rate

KL_DIVERGENCE_MIN = 0.01
KL_DIVERGENCE_MAX = 0.05
EARLY_STOP_ON_TOOL_TASKS_MAX = 0.05


def tail_ref_kl_mean(ref_kls: list[float]) -> float:
    if len(ref_kls) < 3:
        return 0.0
    return float(np.mean(np.abs(ref_kls[-3:])))


def kl_in_gate_band(mean_kl: float) -> bool:
    return KL_DIVERGENCE_MIN <= mean_kl <= KL_DIVERGENCE_MAX


def early_stop_ok(controller: MonolithicController) -> bool:
    probe_rate = probe_premature_stop_rate(controller)["premature_stop_rate"]
    tool_rate = early_stop_tool_rate(controller)
    return (
        probe_rate <= EARLY_STOP_ON_TOOL_TASKS_MAX
        and tool_rate <= EARLY_STOP_ON_TOOL_TASKS_MAX
    )


def run_ppo_until_kl_band(
    trainer: PPOTrainer,
    controller: MonolithicController,
    build_batch,
    train_examples: list[dict],
    reward_fn,
    *,
    min_epochs: int = 6,
    max_epochs: int = 20,
) -> tuple[list[PPOStats], list[float]]:
    """
    Run PPO until tail |ref_kl| is in [0.01, 0.05] without breaking early-stop probes.

    Returns per-epoch stats and ref_kl history.
    """
    ref_kls: list[float] = []
    stats_history: list[PPOStats] = []
    lr = trainer.config.learning_rate_stop_head

    for epoch in range(max_epochs):
        batch = build_batch(train_examples, controller, reward_fn)
        stats = trainer.update(batch)
        stats_history.append(stats)
        ref_kls.append(stats.ref_kl)

        mean_kl = tail_ref_kl_mean(ref_kls)
        if epoch + 1 >= min_epochs and kl_in_gate_band(mean_kl) and early_stop_ok(controller):
            break

        if epoch + 1 >= min_epochs and mean_kl < KL_DIVERGENCE_MIN:
            lr = min(lr * 1.6, 0.02)
            trainer.config.learning_rate_stop_head = lr
        elif mean_kl > KL_DIVERGENCE_MAX:
            lr = max(lr * 0.5, 1e-5)
            trainer.config.learning_rate_stop_head = lr
            if not early_stop_ok(controller):
                break

    return stats_history, ref_kls


def drift_kl_into_band(
    trainer: PPOTrainer,
    controller: MonolithicController,
    build_batch,
    train_examples: list[dict],
    reward_fn,
    *,
    max_epochs: int = 10,
) -> tuple[list[PPOStats], list[float]]:
    """KL-penalty-free stop-head drift; hidden layers stay frozen."""
    ref_kls: list[float] = []
    stats_history: list[PPOStats] = []
    saved_kl_coef = trainer._kl_coef
    trainer._kl_coef = 0.0
    trainer.config.freeze_hidden = True
    lr = 0.004

    for _ in range(max_epochs):
        trainer.config.learning_rate_stop_head = lr
        batch = build_batch(train_examples, controller, reward_fn)
        stats = trainer.update(batch)
        stats_history.append(stats)
        ref_kls.append(stats.ref_kl)

        mean_kl = tail_ref_kl_mean(ref_kls)
        if len(ref_kls) >= 3 and kl_in_gate_band(mean_kl) and early_stop_ok(controller):
            break
        if not early_stop_ok(controller):
            break
        if len(ref_kls) >= 2 and mean_kl < KL_DIVERGENCE_MIN:
            lr = min(lr * 1.3, 0.012)

    trainer._kl_coef = saved_kl_coef
    return stats_history, ref_kls
