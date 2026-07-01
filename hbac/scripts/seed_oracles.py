from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.mock import MockEnv
from hbac.envs.tau_bench import TauBenchEnv
from hbac.envs.toolbench import ToolBenchEnv
from hbac.scripts.export_sft import export_grpo_groups, export_sft

app = typer.Typer(help="Generate seed oracle trajectories without API keys (Phase 1)")


class ScriptedLLM(LLMBackend):
    """Deterministic LLM for seed data generation."""

    SCRIPTS = {
        "mock": [
            '{"tool_name": "bash", "tool_input": "explore"}',
            '{"tool_name": "submit", "tool_input": "4"}',
        ],
        "lcb": [
            '{"tool_name": "generate_code", "tool_input": "a=int(input())\\nb=int(input())\\nprint(a+b)"}',
            '{"tool_name": "run_tests", "tool_input": ""}',
        ],
    }

    def __init__(self, script_key: str) -> None:
        super().__init__(LLMConfig())
        self.script = self.SCRIPTS[script_key]
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


@app.command()
def main(
    output: str = typer.Option("data/oracles/seed", help="Output directory"),
    include_tool_tau: bool = typer.Option(True, help="Also seed toolbench/tau_bench oracle pools"),
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runner_cfg = RunnerConfig(max_steps=5, output_dir=out_dir)
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")

    configs = [
        ("mock_success_1", MockEnv(), "mock-1", "prompt"),
        ("mock_success_2", MockEnv(), "mock-2", "prompt"),
        (
            "lcb",
            LiveCodeBenchEnv(local_mode=True),
            "lcb-local-1",
            ReActRunner.system_prompt_for_benchmark("livecodebench"),
        ),
    ]

    scripts = {
        "mock_success_1": [
            '{"tool_name": "bash", "tool_input": "explore"}',
            '{"tool_name": "submit", "tool_input": "4"}',
        ],
        "mock_success_2": [
            '{"tool_name": "submit", "tool_input": "hello world"}',
        ],
        "lcb": [
            '{"tool_name": "generate_code", "tool_input": "a=int(input())\\nb=int(input())\\nprint(a+b)"}',
            '{"tool_name": "run_tests", "tool_input": ""}',
        ],
    }

    count = 0
    for script_key, env, task_id, prompt in configs:
        llm = ScriptedLLM.__new__(ScriptedLLM)
        LLMBackend.__init__(llm, LLMConfig())
        llm.script = scripts[script_key]
        llm.i = 0
        runner = ReActRunner(llm, runner_cfg)
        traj = runner.run_episode(env, prompt, task_id)
        oracle_store.append(traj)
        if traj.success:
            count += 1
        typer.echo(f"{task_id}: success={traj.success}")

    export_sft(out_dir / "oracles.jsonl", out_dir / "sft.jsonl")
    export_grpo_groups(out_dir / "oracles.jsonl", out_dir / "grpo_groups.jsonl")

    typer.echo(f"Seed oracles: {count}/{len(configs)} successful -> {out_dir}")
    typer.echo("Phase 1 seed data ready for Phase 2 training subset.")

    if include_tool_tau:
        _seed_domain_pool("toolbench", ToolBenchEnv, output)
        _seed_domain_pool("tau_bench", TauBenchEnv, output)


def _seed_domain_pool(benchmark: str, env_cls, oracle_root: str) -> None:
    """Write successful stub trajectories into data/oracles/<benchmark>/<run_id>/."""
    from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(oracle_root) / benchmark / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg = RunnerConfig(max_steps=6, output_dir=out_dir)
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")

    episodes = [ep for ep in DETERMINISTIC_EPISODES if ep.env_name == benchmark]
    scripts = {ep.task_id: ep.responses for ep in episodes}
    prompts = {ep.task_id: ep.system_prompt for ep in episodes}

    count = 0
    for task_id, script in scripts.items():
        llm = ScriptedLLM.__new__(ScriptedLLM)
        LLMBackend.__init__(llm, LLMConfig())
        llm.script = script
        llm.i = 0
        runner = ReActRunner(llm, runner_cfg)
        env = env_cls(budget_tokens=50_000)
        traj = runner.run_episode(env, prompts[task_id], task_id)
        oracle_store.append(traj)
        if traj.success:
            count += 1
        typer.echo(f"{benchmark}/{task_id}: success={traj.success}")

    export_sft(out_dir / "oracles.jsonl", out_dir / "sft.jsonl")
    typer.echo(f"{benchmark} pool: {count}/{len(scripts)} -> {out_dir}")


if __name__ == "__main__":
    app()
