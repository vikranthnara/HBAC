from __future__ import annotations

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.baselines.ref_orc import ReFORCRunner
from hbac.baselines.tab import TABRunner
from hbac.core.config import LLMConfig, LiveCodeBenchConfig, ReFORCConfig, RunConfig, SWEBenchConfig, TABConfig
from hbac.core.llm import LLMBackend
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.mock import MockEnv
from hbac.envs.swe_bench import SWEBenchEnv

app = typer.Typer(help="Run HBAC baselines on benchmarks")


def _make_runner(baseline: str, llm: LLMBackend, run_cfg: RunConfig):
    runner_cfg = RunnerConfig(
        max_steps=run_cfg.max_steps,
        output_dir=__import__("pathlib").Path(run_cfg.output_dir),
    )
    if baseline == "react":
        return ReActRunner(llm, runner_cfg)
    if baseline == "tab":
        return TABRunner(llm, runner_cfg, TABConfig())
    if baseline == "ref_orc":
        return ReFORCRunner(llm, runner_cfg, ReFORCConfig())
    raise typer.BadParameter(f"Unknown baseline: {baseline}")


def _env_factory(env_name: str, budget: int, local_mode: bool = False):
    if env_name == "mock":
        return lambda: MockEnv(budget_tokens=budget)
    if env_name == "swe_bench":
        return lambda: SWEBenchEnv(
            budget_tokens=budget,
            config=SWEBenchConfig(),
            local_mode=local_mode,
        )
    if env_name == "livecodebench":
        return lambda: LiveCodeBenchEnv(
            budget_tokens=budget,
            config=LiveCodeBenchConfig(),
            local_mode=local_mode,
        )
    raise typer.BadParameter(f"Unknown env: {env_name}")


def _list_tasks(env_name: str, limit: int | None, local_mode: bool) -> list[str]:
    if env_name == "mock":
        return list(MockEnv.TASKS.keys())[: limit or None]
    if env_name == "swe_bench":
        try:
            return SWEBenchEnv.list_task_ids(limit=limit)
        except Exception:
            return ["swe-local-1"][: limit or 1]
    if env_name == "livecodebench":
        if local_mode:
            ids = list(LiveCodeBenchEnv(local_mode=True)._problems.keys())
        else:
            try:
                ids = LiveCodeBenchEnv.list_task_ids(limit=limit)
            except Exception:
                ids = list(LiveCodeBenchEnv(local_mode=True)._problems.keys())
        return ids[:limit] if limit else ids
    raise typer.BadParameter(f"Unknown env: {env_name}")


@app.command()
def main(
    baseline: str = typer.Option("react", help="react | tab | ref_orc"),
    env: str = typer.Option("mock", help="mock | swe_bench | livecodebench"),
    model: str = typer.Option("auto", help="provider:model or auto (uses FreeLLMAPI when configured)"),
    budget: int = typer.Option(50_000, help="Global token budget per task"),
    output: str = typer.Option("results", help="Output directory"),
    limit: int | None = typer.Option(None, help="Max tasks to run"),
    local_mode: bool = typer.Option(False, help="Use local fallback tasks (no dataset/docker)"),
    max_steps: int = typer.Option(100, help="Max agent steps per episode"),
) -> None:
    llm = LLMBackend.from_spec(model)
    run_cfg = RunConfig(budget_tokens=budget, output_dir=output, limit=limit, max_steps=max_steps)
    runner = _make_runner(baseline, llm, run_cfg)

    task_ids = _list_tasks(env, limit, local_mode)
    if not task_ids:
        typer.echo("No tasks found.")
        raise typer.Exit(1)

    system_prompt = ReActRunner.system_prompt_for_benchmark(
        "livecodebench" if env == "livecodebench" else "swe_bench"
    )
    if env == "mock":
        system_prompt = "Respond with JSON: {\"tool_name\": \"bash|submit\", \"tool_input\": \"...\"}"

    factory = _env_factory(env, budget, local_mode=local_mode or env == "mock")
    metrics = runner.run_batch(factory, task_ids, system_prompt)
    summary = metrics.summarize()
    typer.echo(f"Pass@1: {summary['pass_at_1']:.2%}")
    typer.echo(f"Budget violation rate: {summary['budget_violation_rate']:.2%}")
    typer.echo(f"Mean tokens: {summary['mean_tokens']:.0f}")


if __name__ == "__main__":
    app()
