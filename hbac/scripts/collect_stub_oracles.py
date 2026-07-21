"""Collect deterministic stub oracles for live-eval benchmarks (toolbench, tau_bench, mock, swe)."""

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
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES, make_env
from hbac.scripts.export_sft import export_grpo_groups, export_sft

app = typer.Typer(help="Expand stub oracles for live-eval benchmark alignment")


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


BENCHMARKS = frozenset({"toolbench", "tau_bench", "mock", "swe_bench"})


@app.command()
def main(
    output: str = typer.Option("data/oracles/stub_live", help="Output root"),
    benchmarks: str = typer.Option(
        "toolbench,tau_bench,mock,swe_bench",
        help="Comma-separated stub benchmarks",
    ),
) -> None:
    wanted = {b.strip() for b in benchmarks.split(",") if b.strip()}
    unknown = wanted - BENCHMARKS
    if unknown:
        raise typer.BadParameter(f"Unsupported benchmarks: {unknown}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(output) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    runner_cfg = RunnerConfig(max_steps=8, output_dir=out_root)

    total = 0
    success = 0
    for ep in DETERMINISTIC_EPISODES:
        if ep.env_name not in wanted:
            continue
        if ep.env_name == "mock":
            env = MockEnv(budget_tokens=50_000)
            task_id = ep.task_id if ep.task_id.startswith("mock") else "mock-1"
        else:
            env = make_env(ep.env_name, budget=50_000)
            task_id = ep.task_id

        bench_dir = out_root / ep.env_name
        bench_dir.mkdir(parents=True, exist_ok=True)
        oracle_store = TrajectoryStore(bench_dir / "oracles.jsonl")

        llm = ScriptedLLM(ep.responses)
        prompt = ep.system_prompt or ReActRunner.system_prompt_for_benchmark(ep.env_name)
        traj = ReActRunner(llm, runner_cfg).run_episode(env, prompt, task_id)
        oracle_store.append(traj)
        total += 1
        if traj.success:
            success += 1
        typer.echo(f"{ep.env_name}/{task_id}: success={traj.success}")

    # Merge per-benchmark pools into run-level oracles.jsonl for training
    merged = TrajectoryStore(out_root / "oracles.jsonl")
    for bench in sorted(wanted):
        bench_oracles = out_root / bench / "oracles.jsonl"
        if bench_oracles.is_file():
            for traj in TrajectoryStore(bench_oracles).load_all():
                merged.append(traj)

    export_sft(out_root / "oracles.jsonl", out_root / "sft.jsonl")
    export_grpo_groups(out_root / "oracles.jsonl", out_root / "grpo_groups.jsonl")
    typer.echo(f"Stub live oracles: {success}/{total} successful -> {out_root}")


if __name__ == "__main__":
    app()
