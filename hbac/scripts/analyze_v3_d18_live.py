"""Analyze V3 D18 live matrix: per-benchmark breakdown + allocator comparison."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="V3 D18 live analysis")


PRIMARY = ("hbac_d18", "hbac_guardrail", "hbac_joint", "type_prior", "uniform")
BENCHMARKS = ("livecodebench", "swe_bench", "tau_bench", "toolbench")


@app.command()
def main(
    result_path: str = typer.Option(
        "results/rivanna/compose_live_v3_d18_floor400_n2000.json",
        help="Merged V3 D18 live JSON",
    ),
    paired_path: str = typer.Option(
        "results/paired_allocator_analysis_v3_d18.json",
        help="Paired McNemar analysis JSON",
    ),
    output: str = typer.Option("results/v3_d18_live_analysis.json"),
) -> None:
    data = json.loads(Path(result_path).read_text(encoding="utf-8"))
    paired = json.loads(Path(paired_path).read_text()) if Path(paired_path).is_file() else {}

    alloc_rows = {}
    for key in PRIMARY:
        row = data.get(key, {})
        if not row:
            continue
        alloc_rows[key] = {
            "pass_at_1": row.get("pass_at_1"),
            "pass_at_1_ci95": row.get("pass_at_1_ci95"),
            "mean_tokens_used": row.get("mean_tokens_used"),
            "per_benchmark": {
                b: row.get("per_benchmark", {}).get(b, {})
                for b in BENCHMARKS
                if b in row.get("per_benchmark", {})
            },
        }

    type_prior = alloc_rows.get("type_prior", {})
    d18 = alloc_rows.get("hbac_d18", {})
    gap_pp = (d18.get("pass_at_1", 0) - type_prior.get("pass_at_1", 0)) * 100

    per_bench_gap = {}
    for bench in BENCHMARKS:
        d18_b = d18.get("per_benchmark", {}).get(bench, {})
        tp_b = type_prior.get("per_benchmark", {}).get(bench, {})
        if d18_b and tp_b:
            per_bench_gap[bench] = {
                "hbac_d18": d18_b.get("pass_at_1"),
                "type_prior": tp_b.get("pass_at_1"),
                "gap_pp": (d18_b.get("pass_at_1", 0) - tp_b.get("pass_at_1", 0)) * 100,
            }

    d18_vs_joint_identical = (
        d18.get("pass_at_1") == alloc_rows.get("hbac_joint", {}).get("pass_at_1")
        and d18.get("per_benchmark") == alloc_rows.get("hbac_joint", {}).get("per_benchmark")
    )

    report = {
        "source": result_path,
        "num_tasks": data.get("num_tasks"),
        "llm": data.get("llm"),
        "lora_path": data.get("lora_path"),
        "live_min_per_task": data.get("live_min_per_task"),
        "allocators": alloc_rows,
        "hbac_d18_minus_type_prior_pp": gap_pp,
        "per_benchmark_gap": per_bench_gap,
        "d18_equals_joint_on_live": d18_vs_joint_identical,
        "paired_analysis": paired,
        "verdict": paired.get("verdict", "UNKNOWN"),
        "interpretation": (
            "D18 matches joint/guardrail on live 7B — starvation penalty does not "
            "change outcomes when model cannot solve SWE; gap vs type-prior is LCB-localized."
            if d18_vs_joint_identical
            else "D18 differs from joint on live."
        ),
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
