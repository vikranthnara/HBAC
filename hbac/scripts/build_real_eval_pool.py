"""Merge real LCB + SWE-bench Lite oracles into unified eval pool (replaces stubs)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

from hbac.core.trajectory import TrajectoryStore
from hbac.training.dataset import find_oracle_paths

app = typer.Typer(help="Build real_eval oracle pool: LCB bulk + SWE Lite + existing tool/tau")


@app.command()
def main(
    output: str = typer.Option("data/oracles/real_eval", help="Merged pool root"),
    lcb_problems: int = typer.Option(500, help="LCB problems via bulk_expand"),
    swe_limit: int = typer.Option(50, help="SWE-bench Lite instances"),
    include_stub_tool: bool = typer.Option(
        True, help="Include existing toolbench/tau oracles from data/oracles"
    ),
    skip_lcb_expand: bool = typer.Option(False, help="Skip bulk_expand if lcb exists"),
) -> None:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(output) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    merged = TrajectoryStore(out_root / "oracles.jsonl")

    # 1) LCB bulk
    lcb_dirs = sorted(Path("data/oracles/bulk").glob("*/oracles.jsonl"), reverse=True)
    if not lcb_dirs or not skip_lcb_expand:
        typer.echo(f"Expanding {lcb_problems} LCB problems...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.bulk_expand_oracles",
                "--num-problems",
                str(lcb_problems),
                "--output",
                "data/oracles/bulk",
            ],
            check=True,
        )
        lcb_dirs = sorted(Path("data/oracles/bulk").glob("*/oracles.jsonl"), reverse=True)

    if lcb_dirs:
        for traj in TrajectoryStore(lcb_dirs[0]).load_successful():
            merged.append(traj)
        typer.echo(f"LCB: {len(TrajectoryStore(lcb_dirs[0]).load_successful())} oracles")

    # 2) SWE Lite
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hbac.scripts.collect_swe_lite_oracles",
            "--limit",
            str(swe_limit),
            "--output",
            "data/oracles/swe_lite",
        ],
        check=False,
    )
    swe_dirs = sorted(Path("data/oracles/swe_lite").glob("*/oracles.jsonl"), reverse=True)
    if swe_dirs:
        for traj in TrajectoryStore(swe_dirs[0]).load_successful():
            merged.append(traj)
        typer.echo(f"SWE Lite: {len(TrajectoryStore(swe_dirs[0]).load_successful())} oracles")

    # 3) Existing real tool/tau pools
    if include_stub_tool:
        for path in find_oracle_paths(Path("data/oracles")):
            if "stub_live" in str(path) or "real_eval" in str(path):
                continue
            for traj in TrajectoryStore(path).load_successful():
                if traj.benchmark in {"toolbench", "tau_bench", "mock"}:
                    merged.append(traj)

    # 4) Per-benchmark symlinks for discovery
    by_bench: dict[str, list] = {}
    for traj in TrajectoryStore(out_root / "oracles.jsonl").load_all():
        by_bench.setdefault(traj.benchmark, []).append(traj)

    manifest = {
        "run_id": run_id,
        "output": str(out_root),
        "counts": {b: len(v) for b, v in by_bench.items()},
        "total": sum(len(v) for v in by_bench.values()),
        "note": "Real LCB (bulk_expand) + SWE Lite golden-patch oracles; not stub swe-local",
    }
    for bench, trajs in by_bench.items():
        bench_dir = out_root / bench
        bench_dir.mkdir(exist_ok=True)
        store = TrajectoryStore(bench_dir / "oracles.jsonl")
        for t in trajs:
            store.append(t)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Latest symlink
    latest = Path(output) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out_root.name)
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
