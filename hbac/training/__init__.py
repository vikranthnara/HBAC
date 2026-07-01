"""Phase 2 training: Variant A monolithic controller, PPO, reward validation."""

from hbac.training.config import PPOConfig, VariantAConfig
from hbac.training.controller import MonolithicController
from hbac.training.reward import TaskControllerReward
from hbac.training.validation import all_passed, run_all_validations

__all__ = [
    "PPOConfig",
    "VariantAConfig",
    "MonolithicController",
    "TaskControllerReward",
    "all_passed",
    "run_all_validations",
]
