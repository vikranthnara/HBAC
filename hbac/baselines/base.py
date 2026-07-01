from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hbac.core.env import AgentEnv
from hbac.core.llm import LLMBackend
from hbac.core.metrics import MetricsLogger
from hbac.core.trajectory import TrajectoryStore
from hbac.core.types import AgentAction, Observation, Trajectory, TrajectoryStep


@dataclass
class RunnerConfig:
    max_steps: int = 100
    max_tokens_per_step: int = 4096
    output_dir: Path = field(default_factory=lambda: Path("results"))


class BaseRunner(ABC):
    name: str = "base"

    def __init__(
        self,
        llm: LLMBackend,
        config: RunnerConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or RunnerConfig()

    @abstractmethod
    def max_tokens_for_step(self, obs: Observation, turn: int) -> int: ...

    @abstractmethod
    def should_stop_early(
        self, obs: Observation, turn: int, llm_text: str, step_tokens: int = 0
    ) -> bool: ...

    def on_step_complete(
        self,
        obs: Observation,
        turn: int,
        action: AgentAction,
        llm_text: str,
        step_tokens: int = 0,
    ) -> None:
        """Hook for baselines that track per-turn state (TAB, Re-FORC)."""

    def build_messages(self, obs: Observation, system_prompt: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(obs.history)
        if obs.env_feedback and (not obs.history or obs.history[-1]["content"] != obs.env_feedback):
            messages.append({"role": "user", "content": obs.env_feedback})
        return messages

    def parse_action(self, text: str) -> AgentAction:
        text = text.strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return AgentAction(
                    thought=data.get("thought"),
                    tool_name=data.get("tool_name", data.get("action", "bash")),
                    tool_input=data.get("tool_input", data.get("command", data.get("input"))),
                    stop=bool(data.get("stop", False)),
                )
            except json.JSONDecodeError:
                pass

        thought = None
        action_text = text
        if "Action:" in text:
            parts = text.split("Action:", 1)
            thought = parts[0].replace("Thought:", "").strip()
            action_text = parts[1].strip()

        if action_text.lower().startswith("submit"):
            return AgentAction(tool_name="submit", tool_input="", stop=True, thought=thought)

        return AgentAction(
            thought=thought,
            tool_name="bash",
            tool_input=action_text,
            stop=False,
        )

    def run_episode(self, env: AgentEnv, system_prompt: str, task_id: str) -> Trajectory:
        obs = env.reset(task_id)
        steps: list[TrajectoryStep] = []
        total_llm_tokens = 0
        done = False
        turn = 0

        while not done and turn < self.config.max_steps:
            if env.remaining_budget <= 0:
                break

            max_tokens = min(
                self.max_tokens_for_step(obs, turn),
                max(env.remaining_budget, 1),
                self.config.max_tokens_per_step,
            )

            messages = self.build_messages(obs, system_prompt)
            response = self.llm.complete(messages, max_tokens=max_tokens)
            allowed = min(response.total_tokens, max(env.remaining_budget, 0))
            total_llm_tokens += allowed
            if allowed > 0:
                env.record_llm_tokens(allowed)

            if self.should_stop_early(obs, turn, response.text, response.total_tokens):
                action = AgentAction(tool_name="submit", tool_input="", stop=True)
            else:
                action = self.parse_action(response.text)
                action.max_tokens = max_tokens

            self.on_step_complete(obs, turn, action, response.text, response.total_tokens)
            step_result = env.step(action)
            obs = step_result.obs
            done = step_result.done

            steps.append(
                TrajectoryStep(
                    turn=turn,
                    state_summary=f"turn={turn} budget_left={env.remaining_budget}",
                    action=action,
                    observation=obs.env_feedback,
                    tokens=response.total_tokens,
                    budget_allocated=max_tokens,
                    llm_response=response.text,
                    metadata={
                        "latency_ms": response.latency_ms,
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                    },
                )
            )
            turn += 1

        eval_result = env.evaluate()
        return Trajectory(
            task_id=env.task_spec.task_id,
            benchmark=env.task_spec.benchmark,
            model=f"{self.llm.config.provider}:{self.llm.config.model}",
            baseline=self.name,
            success=eval_result.success,
            total_tokens=eval_result.total_tokens or total_llm_tokens,
            budget=env.task_spec.budget_tokens,
            steps=steps,
            final_output=eval_result.final_output,
            metadata={
                "test_output": eval_result.test_output,
                "budget_violated": eval_result.budget_violated,
                **eval_result.metadata,
            },
        )

    def run_batch(
        self,
        env_factory,
        task_ids: list[str],
        system_prompt: str,
        run_id: str | None = None,
    ) -> MetricsLogger:
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = self.config.output_dir / self.name / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = MetricsLogger()
        traj_store = TrajectoryStore(out_dir / "trajectories.jsonl")

        for task_id in task_ids:
            env = env_factory()
            trajectory = self.run_episode(env, system_prompt, task_id)
            metrics.record(
                task_id=trajectory.task_id,
                success=trajectory.success,
                total_tokens=trajectory.total_tokens,
                budget=trajectory.budget,
                baseline=self.name,
            )
            traj_store.append(trajectory)

        metrics.save(out_dir / "metrics.json")
        return metrics
