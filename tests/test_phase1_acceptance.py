from __future__ import annotations

from pathlib import Path

import pytest

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.baselines.ref_orc import ReFORCRunner
from hbac.baselines.tab import TABRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.mock import MockEnv
from hbac.envs.swe_bench import SWEBenchEnv
from hbac.scripts.export_sft import export_grpo_groups, export_sft
from hbac.scripts.seed_oracles import ScriptedLLM


class MockLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=10, latency_ms=1.0)


class TestPhase1Acceptance:
    """End-to-end Phase 1 acceptance criteria."""

    def test_all_baselines_run_on_mock(self):
        responses = [
            '{"tool_name": "bash", "tool_input": "x"}',
            '{"tool_name": "submit", "tool_input": "4"}',
        ]
        cfg = RunnerConfig(max_steps=5, output_dir=Path("/tmp/hbac_test"))
        for cls in (ReActRunner, TABRunner, ReFORCRunner):
            traj = cls(MockLLM(responses), cfg).run_episode(MockEnv(), "p", "mock-1")
            assert traj.steps

    def test_env_wrappers_instantiate(self):
        assert SWEBenchEnv(local_mode=True).reset("swe-local-1")
        assert LiveCodeBenchEnv(local_mode=True).reset("lcb-local-1")
        from hbac.envs.toolbench import ToolBenchEnv
        from hbac.envs.tau_bench import TauBenchEnv

        assert ToolBenchEnv().reset("toolbench-local-1")
        assert TauBenchEnv().reset("tau-local-1")

    def test_oracle_export_pipeline(self, tmp_path):
        runner = ReActRunner(
            MockLLM(
                [
                    '{"tool_name": "bash", "tool_input": "x"}',
                    '{"tool_name": "submit", "tool_input": "4"}',
                ]
            ),
            RunnerConfig(max_steps=5, output_dir=tmp_path),
        )
        traj = runner.run_episode(MockEnv(), "p", "mock-1")
        store = TrajectoryStore(tmp_path / "oracles.jsonl")
        store.append(traj)
        assert export_sft(tmp_path / "oracles.jsonl", tmp_path / "sft.jsonl") >= 1
        assert export_grpo_groups(tmp_path / "oracles.jsonl", tmp_path / "grpo.jsonl") >= 1

    def test_seed_oracle_generation(self, tmp_path):
        out = tmp_path / "seed"
        runner_cfg = RunnerConfig(max_steps=5, output_dir=out)
        store = TrajectoryStore(out / "oracles.jsonl")
        traj = ReActRunner(ScriptedLLM("mock"), runner_cfg).run_episode(MockEnv(), "p", "mock-1")
        store.append(traj)
        assert traj.success
        assert export_sft(out / "oracles.jsonl", out / "sft.jsonl") == 1
