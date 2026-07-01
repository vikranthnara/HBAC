from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PPOConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HBAC_PPO_", extra="ignore")

    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    kl_coef: float = 0.02
    kl_target: float = 0.01
    kl_adaptive: bool = True
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_epochs: int = 4
    batch_size: int = 32
    max_grad_norm: float = 0.5
    freeze_hidden: bool = True
    learning_rate_stop_head: float = 1e-4


class VariantAConfig(BaseSettings):
    """Phase 2: monolithic Level-2 controller (stop head only, Stage 1 curriculum)."""

    model_config = SettingsConfigDict(env_prefix="HBAC_VariantA_", extra="ignore")

    stage: int = 1  # 1=stop only, 2=+tools, 3=+approx
    hidden_dim: int = 128
    max_subset_size: int = 50
    checkpoint_dir: str = "checkpoints/variant_a"
    ppo: PPOConfig = Field(default_factory=PPOConfig)
