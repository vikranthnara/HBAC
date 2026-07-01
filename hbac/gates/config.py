"""Thresholds for Phase 1/2 Go/No-Go gates (Research Plan §16)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase1Thresholds:
    env_execution_success_rate: float = 1.0
    oracle_yield_rate: float = 0.60
    min_oracle_trajectories: int = 500
    max_oracle_trajectories: int = 1000
    pomdp_parse_compliance: float = 1.0
    baseline_val_samples: int = 100
    baseline_pass_at_1_min: float = 0.40  # floor for easy/medium local split sanity


@dataclass(frozen=True)
class Phase2Thresholds:
    # Stage 1: stop-head only; full a_tool/a_approx gated in Phase 3
    stop_format_compliance: float = 0.95
    early_stop_on_tool_tasks_max: float = 0.05
    budget_violation_rate_max: float = 0.02
    kl_divergence_min: float = 0.01
    kl_divergence_max: float = 0.05
    kl_spike_fail: float = 0.10
    draft_overhead_max: float = 0.15


@dataclass(frozen=True)
class Phase3GatewayThresholds:
    dummy_batch_episodes: int = 10
    dummy_batch_max_seconds: float = 300.0
    overfit_samples: int = 30
    min_reward_improvement: float = 0.0


PHASE1 = Phase1Thresholds()
PHASE2 = Phase2Thresholds()
PHASE3 = Phase3GatewayThresholds()
