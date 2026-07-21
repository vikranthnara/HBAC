"""Summarize live compose eval: bootstrap CIs, per-benchmark, allocator deltas (W1/W11)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer

app = typer.Typer(help="Analyze live compose JSON reports")


def _bootstrap_ci(successes: list[bool], *, n_boot: int = 2000, seed: int = 42) -> dict:
    if not successes:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    n = len(successes)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [successes[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return {"mean": sum(successes) / n, "ci_low": lo, "ci_high": hi, "n": n}


def _analyze_report(data: dict, *, label: str) -> dict:
    out: dict = {"label": label, "llm": data.get("llm"), "lora_path": data.get("lora_path")}
    allocators = {}
    for name in ("uniform", "clear_compose", "hbac_joint"):
        row = data.get(name, {})
        n = int(row.get("num_tasks") or 0)
        p = float(row.get("pass_at_1") or 0)
        allocators[name] = {
            "pass_at_1": p,
            "mean_batch_reward": row.get("mean_batch_reward"),
            "mean_tokens_used": row.get("mean_tokens_used"),
            "mean_parse_failures_per_task": row.get("mean_parse_failures_per_task"),
            "first_step_valid_json_rate": row.get("first_step_valid_json_rate"),
            "per_benchmark": row.get("per_benchmark", {}),
            "bootstrap_pass_at_1": _bootstrap_ci([True] * round(p * n) + [False] * (n - round(p * n))),
        }
    hb = allocators["hbac_joint"]
    uni = allocators["uniform"]
    out["allocators"] = allocators
    out["hbac_vs_uniform"] = {
        "pass_at_1_delta_pp": (hb["pass_at_1"] - uni["pass_at_1"]) * 100,
        "reward_ratio": (hb["mean_batch_reward"] or 0) / max(uni["mean_batch_reward"] or 0, 1e-6),
        "token_savings": (uni["mean_tokens_used"] or 0) - (hb["mean_tokens_used"] or 0),
    }
    return out


@app.command()
def main(
    reports: str = typer.Option(
        "results/rivanna/compose_live_bf040_seed47.json,"
        "results/rivanna/compose_live_bf040_seed47_dpo_v2.json",
        help="Comma-separated live compose JSON paths",
    ),
    output: str = typer.Option("results/live_compose_analysis.json", help="Summary JSON"),
) -> None:
    paths = [Path(p.strip()) for p in reports.split(",") if p.strip()]
    analyses: list[dict] = []
    for path in paths:
        if not path.is_file():
            typer.echo(f"skip missing {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        analyses.append(_analyze_report(data, label=path.name))

    summary = {
        "reports_analyzed": len(analyses),
        "analyses": analyses,
        "diagnosis": (
            "If per_benchmark pass@1 ties across allocators but HBAC reward/tokens differ, "
            "the bottleneck is end-task competence on hard stubs (SWE/tau), not JSON or L1."
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
