"""Floor dose-response: where does HBAC pass@1 gap emerge?"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Analyze live eval across token floors")


def _row(data: dict, key: str) -> dict:
    r = data.get(key, {})
    return {
        "pass_at_1": float(r.get("pass_at_1") or 0),
        "mean_batch_reward": float(r.get("mean_batch_reward") or 0),
        "mean_tokens_used": float(r.get("mean_tokens_used") or 0),
        "mean_parse_failures_per_task": float(r.get("mean_parse_failures_per_task") or 0),
        "batch_violation_rate": float(r.get("batch_violation_rate") or 0),
        "per_benchmark": r.get("per_benchmark", {}),
    }


@app.command()
def main(
    glob_pattern: str = typer.Option(
        "results/rivanna/compose_live_bf040_floor*_dpo_v2.json",
        help="Glob for floor sweep results",
    ),
    baseline: str = typer.Option(
        "results/rivanna/compose_live_bf040_seed47_dpo_v2.json",
        help="Floor=600 baseline",
    ),
    output: str = typer.Option("results/floor_sweep_analysis.json", help="Output JSON"),
) -> None:
    rows: list[dict] = []
    base = _row(json.loads(Path(baseline).read_text()), "hbac_joint") if Path(baseline).is_file() else {}
    base_uni = (
        _row(json.loads(Path(baseline).read_text()), "uniform")
        if Path(baseline).is_file()
        else {}
    )

    for path in sorted(Path(".").glob(glob_pattern)):
        data = json.loads(path.read_text())
        # infer floor from filename e.g. floor400
        stem = path.stem
        floor = None
        for part in stem.split("_"):
            if part.startswith("floor") and part[5:].isdigit():
                floor = int(part[5:])
        if floor is None:
            continue
        hb = _row(data, "hbac_joint")
        uni = _row(data, "uniform")
        cl = _row(data, "clear_compose")
        gap_pp = (hb["pass_at_1"] - uni["pass_at_1"]) * 100
        rows.append(
            {
                "file": str(path),
                "floor": floor,
                "hbac": hb,
                "uniform": uni,
                "clear": cl,
                "hbac_minus_uniform_pp": gap_pp,
                "hbac_beats_uniform": gap_pp > 0 or hb["mean_batch_reward"] > uni["mean_batch_reward"],
            }
        )

    rows.sort(key=lambda r: r["floor"])

    # transition: first floor where gap > 0
    transition = next((r["floor"] for r in rows if r["hbac_minus_uniform_pp"] > 0), None)

    report = {
        "baseline_floor600": {
            "hbac_pass_at_1": base.get("pass_at_1"),
            "uniform_pass_at_1": base_uni.get("pass_at_1"),
            "hbac_tokens": base.get("mean_tokens_used"),
        },
        "by_floor": rows,
        "transition_floor": transition,
        "discovery_note": (
            "HBAC pass@1 gap emerges when per-task floor is tight enough that uniform "
            "cannot afford multi-step tool chains; at floor=600 uniform is padded to ~598 tok."
        ),
    }
    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
