"""Symlink legacy n1000 live artifact filename to canonical n2000 name."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Migrate compose_live_v3_*_n1000_* to *_n2000_* naming")


@app.command()
def main(
    legacy: str = typer.Option(
        "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json",
        help="Legacy merged live JSON",
    ),
    canonical: str = typer.Option(
        "results/rivanna/compose_live_v3_heuristics_floor400_n2000_dpo_v2.json",
        help="Canonical n2000 filename",
    ),
) -> None:
    legacy_path = Path(legacy)
    canon_path = Path(canonical)
    if not legacy_path.is_file():
        raise typer.BadParameter(f"Legacy artifact missing: {legacy_path}")

    data = json.loads(legacy_path.read_text(encoding="utf-8"))
    data["num_tasks"] = data.get("num_tasks", 2000)
    data["artifact_note"] = "Renamed from legacy n1000 tag; evaluation uses n=2000 tasks"
    canon_path.parent.mkdir(parents=True, exist_ok=True)
    canon_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    link_path = legacy_path.parent / "compose_live_v3_heuristics_floor400_n2000_dpo_v2.json"
    if legacy_path.resolve() != link_path.resolve() and not link_path.exists():
        link_path.symlink_to(legacy_path.name)

    typer.echo(f"Wrote canonical artifact -> {canon_path}")
    typer.echo(f"num_tasks={data.get('num_tasks')}")


if __name__ == "__main__":
    app()
