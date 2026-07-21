from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.envs.mock import MockEnv
from hbac.envs.swe_bench import SWEBenchEnv
from hbac.envs.tau_bench import TauBenchEnv
from hbac.envs.toolbench import ToolBenchEnv
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES
from hbac.scripts.run_baseline import _env_factory, _list_tasks

app = typer.Typer(help="Collect oracle trajectories from strong-model or stub rollouts")

STUB_ENVS = frozenset({"toolbench", "tau_bench", "mock"})


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


def _stub_collect(env: str, out_dir: Path, runner_cfg: RunnerConfig, limit: int | None) -> int:
    episodes = [ep for ep in DETERMINISTIC_EPISODES if ep.env_name == env]
    if env == "mock":
        episodes = [
            type("Ep", (), {
                "task_id": "mock-1",
                "system_prompt": "mock",
                "responses": [
                    '{"tool_name": "bash", "tool_input": "explore"}',
                    '{"tool_name": "submit", "tool_input": "4"}',
                ],
            })(),
            type("Ep", (), {
                "task_id": "mock-2",
                "system_prompt": "mock",
                "responses": ['{"tool_name": "submit", "tool_input": "hello world"}'],
            })(),
        ]
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")
    all_store = TrajectoryStore(out_dir / "all_trajectories.jsonl")
    success = 0
    for ep in episodes[: limit or len(episodes)]:
        llm = ScriptedLLM(ep.responses)
        if env == "mock":
            env_instance = MockEnv(budget_tokens=50_000)
        elif env == "toolbench":
            env_instance = ToolBenchEnv(budget_tokens=50_000)
        elif env == "tau_bench":
            env_instance = TauBenchEnv(budget_tokens=50_000)
        elif env == "swe_bench":
            env_instance = SWEBenchEnv(budget_tokens=50_000, local_mode=True)
        else:
            continue
        prompt = getattr(ep, "system_prompt", ReActRunner.system_prompt_for_benchmark(env))
        traj = ReActRunner(llm, runner_cfg).run_episode(env_instance, prompt, ep.task_id)
        all_store.append(traj)
        if traj.success:
            oracle_store.append(traj)
            success += 1
        typer.echo(f"{ep.task_id}: success={traj.success}")
    return success


@app.command()
def main(
    env: str = typer.Option(
        "swe_bench",
        help="swe_bench | livecodebench | toolbench | tau_bench | mock",
    ),
    model: str = typer.Option("auto", help="provider:model, auto, or stub"),
    budget: int = typer.Option(50_000, help="Token budget per task"),
    output: str = typer.Option("data/oracles", help="Output directory"),
    limit: int | None = typer.Option(None, help="Max tasks"),
    local_mode: bool = typer.Option(False, help="Local fallback mode"),
    max_steps: int = typer.Option(100, help="Max steps per episode"),
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / env / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg = RunnerConfig(max_steps=max_steps, output_dir=out_dir)

    if model == "stub" or env in STUB_ENVS:
        success_count = _stub_collect(env, out_dir, runner_cfg, limit)
        total = limit or len([ep for ep in DETERMINISTIC_EPISODES if ep.env_name == env])
        typer.echo(f"Collected {success_count}/{total} stub oracles -> {out_dir / 'oracles.jsonl'}")
        return

    llm = LLMBackend.from_spec(model)
    runner = ReActRunner(llm, runner_cfg)
    task_ids = _list_tasks(env, limit, local_mode)
    factory = _env_factory(env, budget, local_mode=local_mode)
    system_prompt = ReActRunner.system_prompt_for_benchmark(
        "livecodebench" if env == "livecodebench" else env
    )

    all_store = TrajectoryStore(out_dir / "all_trajectories.jsonl")
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")

    success_count = 0
    for task_id in task_ids:
        env_instance = factory()
        trajectory = runner.run_episode(env_instance, system_prompt, task_id)
        all_store.append(trajectory)
        if trajectory.success:
            oracle_store.append(trajectory)
            success_count += 1
        typer.echo(f"{task_id}: success={trajectory.success} tokens={trajectory.total_tokens}")

    typer.echo(f"Collected {success_count}/{len(task_ids)} oracle trajectories -> {oracle_store.path}")


if __name__ == "__main__":
    app()
