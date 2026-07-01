from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hbac.core.types import Observation


def featurize_observation(obs: Observation) -> np.ndarray:
    """Cheap state features for Stage-1 stop controller prototyping."""
    history_len = len(obs.history)
    content_len = sum(len(m.get("content", "")) for m in obs.history)
    feedback_len = len(obs.env_feedback)
    tools = len(obs.tools_available)
    budget_frac = obs.remaining_budget / max(obs.remaining_budget + 1, 1)
    return np.array(
        [
            1.0,
            obs.turn / 100.0,
            history_len / 50.0,
            content_len / 10000.0,
            feedback_len / 5000.0,
            tools / 10.0,
            budget_frac,
        ],
        dtype=np.float64,
    )


@dataclass
class ControllerCheckpoint:
    weights: np.ndarray
    bias: float
    config: dict = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weights=self.weights, bias=self.bias, config=json.dumps(self.config))

    @classmethod
    def load(cls, path: Path) -> ControllerCheckpoint:
        data = np.load(path, allow_pickle=True)
        return cls(
            weights=data["weights"],
            bias=float(data["bias"]),
            config=json.loads(str(data["config"])),
        )


class MonolithicController:
    """
    Variant A Level-2 monolithic controller (Stage 1: a_stop only).

    Logistic stop head over hand-crafted features. PPO trainer updates weights.
    """

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        scale = 0.01
        self.w1 = np.random.randn(input_dim, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim) * scale
        self.b2 = -4.0  # continue-bias prior: ~2% stop prob before training

    def _hidden(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.w1 + self.b1)

    def stop_logit(self, obs: Observation) -> float:
        h = self._hidden(featurize_observation(obs))
        return float(h @ self.w2 + self.b2)

    def stop_prob(self, obs: Observation) -> float:
        return 1.0 / (1.0 + math.exp(-self.stop_logit(obs)))

    def should_stop(self, obs: Observation, threshold: float = 0.5) -> bool:
        return self.stop_prob(obs) >= threshold

    def log_prob_stop(self, obs: Observation, stop: bool) -> float:
        p = min(max(self.stop_prob(obs), 1e-8), 1 - 1e-8)
        return math.log(p if stop else 1 - p)

    def _log_prob_from_features(self, features: np.ndarray, stop: bool) -> float:
        h = np.tanh(features @ self.w1 + self.b1)
        logit = float(h @ self.w2 + self.b2)
        p = 1.0 / (1.0 + math.exp(-logit))
        p = min(max(p, 1e-8), 1 - 1e-8)
        return math.log(p if stop else 1 - p)

    def flat_params(self) -> np.ndarray:
        return np.concatenate(
            [self.w1.ravel(), self.b1.ravel(), self.w2.ravel(), np.array([self.b2])]
        )

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

    def frozen_copy(self) -> MonolithicController:
        other = MonolithicController(self.input_dim, self.hidden_dim)
        other.w1 = np.array(self.w1, copy=True)
        other.b1 = np.array(self.b1, copy=True)
        other.w2 = np.array(self.w2, copy=True)
        other.b2 = float(self.b2)
        return other

    def save(self, path: Path, ref_controller: MonolithicController | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }
        if ref_controller is not None:
            payload["ref_w1"] = ref_controller.w1
            payload["ref_b1"] = ref_controller.b1
            payload["ref_w2"] = ref_controller.w2
            payload["ref_b2"] = np.array([ref_controller.b2])
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path) -> MonolithicController:
        data = np.load(path)
        ctrl = cls(int(data["input_dim"]), int(data["hidden_dim"]))
        ctrl.w1 = data["w1"]
        ctrl.b1 = data["b1"]
        ctrl.w2 = data["w2"]
        ctrl.b2 = float(data["b2"])
        return ctrl
