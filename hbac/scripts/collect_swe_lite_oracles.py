"""Collect SWE-bench Lite oracle trajectories from dataset golden patches."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.core.config import LLMConfig, SWEBenchConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.core.types import Trajectory
from hbac.envs.swe_bench import SWEBenchEnv

app = typer.Typer(help="Build SWE-bench Lite scripted oracles (golden patch replay)")


class GoldenPatchLLM(LLMBackend):
    """Replay minimal ReAct chain ending in golden patch submission."""

    def __init__(self, patch: str) -> None:
        super().__init__(LLMConfig())
        self.patch = patch
        self.i = 0
        self._responses = [
            '{"tool_name": "bash", "tool_input": "ls -la"}',
            json.dumps({"tool_name": "submit", "tool_input": patch[:8000]}),
        ]

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self._responses[min(self.i, len(self._responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=12, completion_tokens=24, latency_ms=1.0)


def _load_lite_instances(limit: int) -> list[dict]:
    cfg = SWEBenchConfig(dataset_name="princeton-nlp/SWE-bench_Lite", split="test")
    env = SWEBenchEnv(config=cfg, local_mode=False, budget_tokens=50_000)
    rows = []
    for iid, row in list(env._instances.items())[:limit]:
        rows.append(dict(row, instance_id=iid))
    return rows


@app.command()
def main(
    limit: int = typer.Option(50, help="Max SWE-bench Lite instances"),
    output: str = typer.Option("data/oracles/swe_lite", help="Output root"),
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_store = TrajectoryStore(out_dir / "oracles.jsonl")
    all_store = TrajectoryStore(out_dir / "all_trajectories.jsonl")

    try:
        instances = _load_lite_instances(limit)
    except Exception as exc:
        typer.echo(f"SWE-bench Lite load failed ({exc}); using local fallback")
        env = SWEBenchEnv(local_mode=True, budget_tokens=50_000)
        instances = list(env._instances.values())[:limit]

    success = 0
    runner_cfg = RunnerConfig(max_steps=6, output_dir=out_dir)
    prompt = ReActRunner.system_prompt_for_benchmark("swe_bench")

    for row in instances:
        iid = row.get("instance_id", row.get("task_id", "unknown"))
        patch = row.get("patch", "") or row.get("test_patch", "")
        if not patch:
            continue
        llm = GoldenPatchLLM(patch)
        env = SWEBenchEnv(local_mode=True, budget_tokens=50_000)
        env._instances[iid] = {
            "instance_id": iid,
            "repo": row.get("repo", "example/repo"),
            "base_commit": row.get("base_commit", "abc"),
            "problem_statement": row.get("problem_statement", "Fix bug"),
            "patch": patch,
        }
        traj = ReActRunner(llm, runner_cfg).run_episode(env, prompt, iid)
        traj.metadata = {
            "repo": row.get("repo"),
            "base_commit": row.get("base_commit"),
            "problem_statement": row.get("problem_statement"),
            "patch": patch,
        }
        all_store.append(traj)
        if traj.success:
            oracle_store.append(traj)
            success += 1
        typer.echo(f"{iid}: success={traj.success} tokens={traj.total_tokens}")

    typer.echo(f"SWE Lite oracles: {success}/{len(instances)} -> {out_dir}")


if __name__ == "__main__":
    app()
