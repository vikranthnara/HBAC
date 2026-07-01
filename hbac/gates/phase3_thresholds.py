"""Phase 3 completion thresholds (3a prototype + 3b LLM scale-up)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase3aThresholds:
    # Allocator mode-collapse: cross-domain budget variance (mixed batches)
    min_domain_allocation_variance: float = 50.0
    # Pareto: policy Pass@1 must exceed uniform stub
    min_pass_at_1_margin: float = 0.0
    min_pass_at_1_floor: float = 0.40
    # Strict budget compliance
    max_batch_violation_rate: float = 0.02
    # Gradient health: fraction of batches with starvation skip
    max_gradient_starvation_rate: float = 0.05
    # Stage 4 joint at 75% budget
    stage4_budget_fraction: float = 0.75


@dataclass(frozen=True)
class Phase3bThresholds:
    min_grpo_samples: int = 4
    max_vram_gb: float = 48.0
    vllm_required: bool = True
    trl_required: bool = True


PHASE3A = Phase3aThresholds()
PHASE3B = Phase3bThresholds()
