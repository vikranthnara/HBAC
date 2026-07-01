from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.core.config import RunConfig
from hbac.core.llm import LLMBackend
from hbac.core.trajectory import TrajectoryStore
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.swe_bench import SWEBenchEnv
from hbac.scripts.run_baseline import _env_factory, _list_tasks

app = typer.Typer(help="Collect oracle trajectories from successful strong-model rollouts")


@app.command()
def main(
    env: str = typer.Option("swe_bench", help="swe_bench | livecodebench"),
    model: str = typer.Option("auto", help="Strong model provider:model or auto"),
    budget: int = typer.Option(50_000, help="Token budget per task"),
    output: str = typer.Option("data/oracles", help="Output directory"),
    limit: int | None = typer.Option(None, help="Max tasks"),
    local_mode: bool = typer.Option(False, help="Local fallback mode"),
    max_steps: int = typer.Option(100, help="Max steps per episode"),
) -> None:
    llm = LLMBackend.from_spec(model)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / env / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = ReActRunner(
        llm,
        RunnerConfig(max_steps=max_steps, output_dir=out_dir),
    )

    task_ids = _list_tasks(env, limit, local_mode)
    factory = _env_factory(env, budget, local_mode=local_mode)
    system_prompt = ReActRunner.system_prompt_for_benchmark(
        "livecodebench" if env == "livecodebench" else "swe_bench"
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
