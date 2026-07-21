"""Summarize capability pilot (uniform only) — SWE gate check."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Analyze capability pilot uniform pass@1 by benchmark")


def _load_pilot_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "result" in data and isinstance(data["result"], dict):
        return {"uniform": data["result"], "llm": data.get("llm")}
    return data


@app.command()
def main(
    source: str = typer.Option(
        "results/rivanna/capability_pilot_uniform.json",
        help="Merged pilot JSON or shard (uniform.json)",
    ),
    swe_gate: float = typer.Option(0.05, help="Minimum SWE pass@1 to proceed"),
    output: str = typer.Option("results/capability_pilot_analysis.json"),
) -> None:
    path = Path(source)
    if not path.is_file():
        shard = Path("results/rivanna/capability_pilot_shards/uniform.json")
        if shard.is_file():
            path = shard
        else:
            raise typer.BadParameter(f"Missing pilot source: {source}")

    data = _load_pilot_payload(path)
    if not data.get("llm"):
        meta_path = path.parent / "meta.json"
        if meta_path.is_file():
            data["llm"] = json.loads(meta_path.read_text(encoding="utf-8")).get("llm")
    uniform = data.get("uniform", data)
    per_bench = uniform.get("per_benchmark", {})
    swe_p = per_bench.get("swe_bench", {}).get("pass_at_1", 0.0)
    lcb_p = per_bench.get("livecodebench", {}).get("pass_at_1", 0.0)

    report = {
        "source": source,
        "llm": data.get("llm"),
        "per_benchmark": per_bench,
        "swe_pass_at_1": swe_p,
        "lcb_pass_at_1": lcb_p,
        "swe_gate": swe_gate,
        "gate_passed": swe_p >= swe_gate,
        "verdict": "PROCEED_TO_ALLOCATOR_STUDY" if swe_p >= swe_gate else "UPGRADE_MODEL_OR_HOLDOUT_SWE",
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
