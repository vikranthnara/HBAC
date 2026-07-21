"""Analyze V3 live n1000+ eval: hbac_fair vs type_prior and full baseline matrix."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="V3 live eval analysis")


@app.command()
def main(
    result_path: str = typer.Option(
        "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json",
        help="Live V3 result JSON",
    ),
    output: str = typer.Option("results/v3_live_analysis.json", help="Analysis output"),
) -> None:
    path = Path(result_path)
    if not path.is_file():
        raise typer.Exit(f"Missing: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    alloc_keys = [
        k
        for k in data
        if isinstance(data[k], dict) and "pass_at_1" in data[k]
    ]

    rows = {}
    for k in sorted(alloc_keys):
        row = data[k]
        rows[k] = {
            "pass_at_1": row.get("pass_at_1"),
            "pass_at_1_ci95": row.get("pass_at_1_ci95"),
            "mean_batch_reward": row.get("mean_batch_reward"),
            "mean_tokens_used": row.get("mean_tokens_used"),
            "batch_violation_rate": row.get("batch_violation_rate"),
            "num_tasks": row.get("num_tasks"),
        }

    hbac_fair = rows.get("hbac_fair", {})
    type_prior = rows.get("type_prior", {})
    hbac = rows.get("hbac_joint", rows.get("hbac", {}))

    fair_p = hbac_fair.get("pass_at_1", 0) or 0
    prior_p = type_prior.get("pass_at_1", 0) or 0
    gap_pp = (fair_p - prior_p) * 100

    report = {
        "source": str(path),
        "num_tasks": data.get("num_tasks") or hbac_fair.get("num_tasks"),
        "live_min_per_task": data.get("live_min_per_task"),
        "fairness_reserve": data.get("fairness_reserve"),
        "hard_min_frac": data.get("hard_min_frac"),
        "hbac_fair_beats_type_prior": fair_p > prior_p,
        "hbac_fair_minus_type_prior_pp": gap_pp,
        "hbac_fair_pass_at_1": fair_p,
        "type_prior_pass_at_1": prior_p,
        "hbac_joint_pass_at_1": hbac.get("pass_at_1"),
        "allocators": rows,
        "verdict": (
            "HBAC_FAIR_WINS"
            if fair_p > prior_p
            else "TIE"
            if abs(gap_pp) < 0.05
            else "TYPE_PRIOR_WINS"
        ),
        "next_steps": [],
    }

    if fair_p <= prior_p:
        report["next_steps"] = [
            "D18: Retrain L1 with fairness penalty in GRPO reward (starvation_rate term)",
            "Tune hard_min_frac sweep (0.10–0.25) on oracle before live",
            "Try scarcity_boost + fairness_reserve combined (D12+D17)",
        ]
    else:
        report["next_steps"] = [
            "Lock v3 live artifact in canonical_artifacts.json",
            "Update Paper v2 §5.4 with live heuristic separation",
            "Submit floor sweep on hbac_fair for dose-response",
        ]

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
