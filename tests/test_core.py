from __future__ import annotations

import pytest

from hbac.baselines.base import BaseRunner, RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.baselines.ref_orc import HeuristicForecaster, ReFORCRunner
from hbac.baselines.tab import HeuristicTABPolicy, StaticTABPolicy, TABRunner
from hbac.core.config import LLMConfig
from hbac.core.cost import BudgetTracker
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.metrics import MetricsLogger
from hbac.core.trajectory import TrajectoryStore
from hbac.core.types import AgentAction, Observation, Trajectory, TrajectoryStep
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.mock import MockEnv
from hbac.envs.swe_bench import SWEBenchEnv


class MockLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.call_count = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return LLMResponse(
            text=text,
            prompt_tokens=10,
            completion_tokens=len(text.split()),
            latency_ms=1.0,
        )


class TestLLMConfig:
    def test_openai_key_from_env(self, monkeypatch):
        monkeypatch.delenv("HBAC_FREELLMAPI_DIR", raising=False)
        monkeypatch.delenv("FREELLMAPI_API_KEY", raising=False)
        monkeypatch.delenv("FREELLMAPI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.delenv("HBAC_LLM_API_KEY", raising=False)
        cfg = LLMConfig(provider="openai")
        assert cfg.api_key == "sk-test-key"
        assert cfg.model == "gpt-4o-mini"

    def test_hbac_prefix_overrides_openai(self, monkeypatch):
        monkeypatch.delenv("HBAC_FREELLMAPI_DIR", raising=False)
        monkeypatch.delenv("FREELLMAPI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-openai")
        monkeypatch.setenv("HBAC_LLM_API_KEY", "sk-from-hbac")
        cfg = LLMConfig(provider="openai")
        assert cfg.api_key == "sk-from-hbac"

    def test_freellmapi_auto_provider(self, monkeypatch):
        monkeypatch.setenv("FREELLMAPI_BASE_URL", "http://127.0.0.1:3001/v1")
        monkeypatch.setenv("FREELLMAPI_API_KEY", "freellmapi-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig()
        assert cfg.provider == "freellmapi"
        assert cfg.api_key == "freellmapi-test"
        assert cfg.base_url == "http://127.0.0.1:3001/v1"
        assert cfg.model == "auto"


class TestBudgetTracker:
    def test_violation(self):
        bt = BudgetTracker(100)
        bt.record(60)
        bt.record(50)
        assert bt.violated
        assert bt.remaining == 0

    def test_hinge_penalty(self):
        bt = BudgetTracker(100)
        bt.record(150)
        assert bt.hinge_penalty(0.01) == pytest.approx(0.5)


class TestMetricsLogger:
    def test_summarize(self):
        m = MetricsLogger()
        m.record("t1", True, 1000, 5000)
        m.record("t2", False, 6000, 5000)
        s = m.summarize()
        assert s["pass_at_1"] == 0.5
        assert s["budget_violation_rate"] == 0.5


class TestTrajectoryStore:
    def test_roundtrip(self, tmp_path):
        store = TrajectoryStore(tmp_path / "t.jsonl")
        traj = Trajectory(
            task_id="t1",
            benchmark="mock",
            model="test",
            baseline="react",
            success=True,
            total_tokens=100,
            budget=1000,
            steps=[
                TrajectoryStep(
                    turn=0,
                    action=AgentAction(tool_name="bash", tool_input="ls"),
                    tokens=50,
                )
            ],
        )
        store.append(traj)
        loaded = store.load_successful()
        assert len(loaded) == 1
        assert loaded[0].task_id == "t1"


class TestReActRunner:
    def test_mock_episode(self):
        llm = MockLLM(
            [
                '{"tool_name": "bash", "tool_input": "explore"}',
                '{"tool_name": "submit", "tool_input": "4"}',
            ]
        )
        runner = ReActRunner(llm, RunnerConfig(max_steps=5))
        env = MockEnv()
        traj = runner.run_episode(env, "test prompt", "mock-1")
        assert traj.task_id == "mock-1"
        assert len(traj.steps) >= 1


class TestTABPolicy:
    def test_static_allocation(self):
        policy = StaticTABPolicy(2048)
        obs = Observation(remaining_budget=5000)
        assert policy.allocate(obs, 0, 5000, 0) == 2048

    def test_heuristic_reserves_budget(self):
        policy = HeuristicTABPolicy()
        obs = Observation(history=[{"role": "user", "content": "x" * 9000}], remaining_budget=3000)
        alloc = policy.allocate(obs, 3, 10000, 7000)
        assert alloc <= 3000


class TestReFORC:
    def test_forecaster_range(self):
        f = HeuristicForecaster()
        obs = Observation(env_feedback="All tests passed")
        psi = f.predict(obs, 1, 1000)
        assert 0.0 <= psi <= 1.0

    def test_early_stop_on_high_confidence(self):
        llm = MockLLM(['{"tool_name": "run_tests", "tool_input": ""}'])
        runner = ReFORCRunner(llm, RunnerConfig(max_steps=5))
        env = LiveCodeBenchEnv(local_mode=True)
        # Will stop early or complete quickly
        traj = runner.run_episode(env, "prompt", "lcb-local-1")
        assert traj.benchmark == "livecodebench"


class TestEnvironments:
    def test_mock_reset_step(self):
        env = MockEnv()
        obs = env.reset("mock-1")
        assert "2+2" in obs.history[0]["content"]
        result = env.step(AgentAction(tool_name="submit", tool_input="4", stop=True))
        assert result.done
        assert env.evaluate().success

    def test_swe_local_mode(self):
        env = SWEBenchEnv(local_mode=True)
        obs = env.reset("swe-local-1")
        assert "problem" in obs.history[0]["content"].lower() or "fix" in obs.history[0]["content"].lower()

    def test_lcb_local_mode(self):
        env = LiveCodeBenchEnv(local_mode=True)
        obs = env.reset("lcb-local-1")
        assert env.task_spec.benchmark == "livecodebench"


class TestExportSFT:
    def test_export_sft(self, tmp_path):
        from hbac.scripts.export_sft import export_sft, export_grpo_groups

        store = TrajectoryStore(tmp_path / "in.jsonl")
        store.append(
            Trajectory(
                task_id="t1",
                benchmark="mock",
                model="m",
                baseline="react",
                success=True,
                total_tokens=10,
                budget=100,
                steps=[
                    TrajectoryStep(
                        turn=0,
                        action=AgentAction(tool_name="bash"),
                        llm_response="hi",
                        observation="ok",
                        budget_allocated=512,
                    )
                ],
            )
        )
        n = export_sft(tmp_path / "in.jsonl", tmp_path / "sft.jsonl")
        assert n == 1
        n2 = export_grpo_groups(tmp_path / "in.jsonl", tmp_path / "grpo.jsonl")
        assert n2 == 1
