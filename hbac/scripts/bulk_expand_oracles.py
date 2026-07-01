"""Generate 500+ diverse scripted oracle trajectories for Go/No-Go volume gates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES

app = typer.Typer(help="Bulk-generate scripted oracle trajectories (no API)")


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


def write_lcb_problems(n: int, path: Path) -> None:
    problems = []
    for i in range(1, n + 1):
        a, b = i, i + 7
        problems.append(
            {
                "question_id": f"lcb-sample-{i}",
                "question_content": f"Read integers {a} and {b} (two lines) and print their sum.",
                "starter_code": "",
                "public_test_cases": json.dumps(
                    [
                        {"input": f"{a}\n{b}", "output": str(a + b), "testtype": "stdin"},
                        {"input": f"{a+1}\n{b+1}", "output": str(a + b + 2), "testtype": "stdin"},
                    ]
                ),
                "metadata": "{}",
            }
        )
    path.write_text(json.dumps(problems, indent=2), encoding="utf-8")


@app.command()
def main(
    num_problems: int = typer.Option(500, help="LCB problems and matching oracle trajectories"),
    output: str = typer.Option("data/oracles/bulk", help="Output root"),
) -> None:
    sample_path = Path(__file__).resolve().parents[1] / "envs" / "data" / "lcb_sample.json"
    write_lcb_problems(num_problems, sample_path)
    typer.echo(f"Wrote {num_problems} problems -> {sample_path}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")
    all_store = TrajectoryStore(out_dir / "all_trajectories.jsonl")

    lcb_script = DETERMINISTIC_EPISODES[1].responses
    prompt = ReActRunner.system_prompt_for_benchmark("livecodebench")
    runner_cfg = RunnerConfig(max_steps=8, output_dir=out_dir)

    env = LiveCodeBenchEnv(local_mode=False)
    task_ids = list(env._problems.keys())
    success = 0

    for tid in task_ids:
        llm = ScriptedLLM(lcb_script)
        traj = ReActRunner(llm, runner_cfg).run_episode(
            LiveCodeBenchEnv(local_mode=False), prompt, tid
        )
        all_store.append(traj)
        if traj.success:
            oracle_store.append(traj)
            success += 1

    typer.echo(f"Bulk oracles: {success}/{len(task_ids)} successful -> {oracle_store.path}")


if __name__ == "__main__":
    app()
