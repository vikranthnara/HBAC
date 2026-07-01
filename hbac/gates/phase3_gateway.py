from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hbac.baselines.base import RunnerConfig
from hbac.baselines.controller import ControllerRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.gates.config import PHASE3
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES, make_env
from hbac.gates.report import GateResult, GateStatus
from hbac.training.controller import MonolithicController
from hbac.training.dataset import find_oracle_paths, load_stop_examples, train_val_split
from hbac.training.probes import probe_premature_stop_rate
from hbac.training.reward import TaskControllerReward
from hbac.training.sft_warmstart import init_continue_bias, sft_warmstart_stop_head
from hbac.scripts.train_variant_a import _build_batch, _eval_accuracy


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = list(responses)
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


def gate_dummy_batch_timing() -> GateResult:
    """10-episode hierarchical dummy batch in <5 minutes, zero crashes."""
    episodes = []
    for i in range(PHASE3.dummy_batch_episodes):
        episodes.append(DETERMINISTIC_EPISODES[i % len(DETERMINISTIC_EPISODES)])

    start = time.perf_counter()
    crashes = 0
    completed = 0
    for ep in episodes:
        try:
            env = make_env(ep.env_name)
            llm = ScriptedLLM(ep.responses)
            runner = ControllerRunner(
                llm,
                MonolithicController(),
                RunnerConfig(max_steps=10, output_dir=Path("/tmp/hbac_gate")),
            )
            runner.run_episode(env, ep.system_prompt, ep.task_id)
            completed += 1
        except Exception:
            crashes += 1

    elapsed = time.perf_counter() - start
    ok_time = elapsed <= PHASE3.dummy_batch_max_seconds
    ok_crash = crashes == 0
    status = GateStatus.PASS if ok_time and ok_crash else GateStatus.FAIL
    return GateResult(
        gate_id="dummy_batch_timing",
        phase="phase3_gateway",
        name="10-episode dummy batch (<5 min, no crashes)",
        status=status,
        measured=elapsed,
        threshold=PHASE3.dummy_batch_max_seconds,
        detail=f"{completed} ok, {crashes} crashes, {elapsed:.1f}s",
    )


def gate_sft_budget_obedience() -> GateResult:
    """Controller episodes respect per-task budgets passed from env."""
    violations = 0
    n = 0
    for ep in DETERMINISTIC_EPISODES:
        budget = 800
        env = make_env(ep.env_name, budget=budget)
        llm = ScriptedLLM(ep.responses)
        runner = ControllerRunner(
            llm,
            MonolithicController(),
            RunnerConfig(max_steps=10, max_tokens_per_step=200, output_dir=Path("/tmp/hbac_gate")),
        )
        traj = runner.run_episode(env, ep.system_prompt, ep.task_id)
        n += 1
        if traj.total_tokens > budget or traj.metadata.get("budget_violated"):
            violations += 1

    rate = violations / n if n else 0.0
    status = GateStatus.PASS if rate <= 0.02 else GateStatus.FAIL
    return GateResult(
        gate_id="sft_budget_obedience",
        phase="phase3_gateway",
        name="SFT/controller budget obedience",
        status=status,
        measured=rate,
        threshold=0.02,
        detail=f"{violations}/{n} episodes violated per-task budget",
    )


def gate_overfit_curve(oracle_root: Path) -> GateResult:
    """30-sample overfit: reward improves without rising premature-stop hacking."""
    from hbac.training.config import PPOConfig
    from hbac.training.ppo import PPOTrainer

    paths = find_oracle_paths(oracle_root)
    examples = load_stop_examples(paths, limit=PHASE3.overfit_samples)
    if len(examples) < 10:
        return GateResult(
            gate_id="overfit_curve",
            phase="phase3_gateway",
            name="30-sample overfit reward curve",
            status=GateStatus.BLOCKED,
            measured=len(examples),
            threshold=PHASE3.overfit_samples,
            detail=f"Only {len(examples)} stop examples available",
        )

    train_ex, _ = train_val_split(examples, val_fraction=0.0, seed=0)
    reward_fn = TaskControllerReward()
    np.random.seed(0)
    controller = MonolithicController()
    init_continue_bias(controller)
    sft_warmstart_stop_head(controller, train_ex, epochs=150)
    trainer = PPOTrainer(
        controller,
        PPOConfig(kl_coef=0.05, kl_adaptive=True, freeze_hidden=True, learning_rate_stop_head=5e-5),
        reward_fn,
    )
    trainer.ref_controller = controller.frozen_copy()

    rewards = []
    for _ in range(8):
        batch = _build_batch(train_ex, controller, reward_fn)
        stats = trainer.update(batch)
        mean_r = float(np.mean([t.reward for t in batch]))
        rewards.append(mean_r)

    improved = rewards[-1] >= rewards[0] + PHASE3.min_reward_improvement
    probe = probe_premature_stop_rate(controller)
    hacking_ok = probe["premature_stop_rate"] <= 0.5  # soft on tiny data

    status = GateStatus.PASS if improved and hacking_ok else GateStatus.FAIL
    return GateResult(
        gate_id="overfit_curve",
        phase="phase3_gateway",
        name="30-sample overfit reward curve",
        status=status,
        measured=rewards[-1] - rewards[0],
        threshold=PHASE3.min_reward_improvement,
        detail=f"reward delta={rewards[-1]-rewards[0]:.3f}, premature_stop={probe['premature_stop_rate']:.1%}",
    )


def run_phase3_gateway_gates(oracle_root: Path) -> list[GateResult]:
    return [
        gate_dummy_batch_timing(),
        gate_sft_budget_obedience(),
        gate_overfit_curve(oracle_root),
    ]
