"""Run feasible Tier-C ablations (H5 draft, H6 counterfactual credit, H7 KL)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(help="Tier C ablation runner (H5, H6, H7)")


def _run(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    output: str = typer.Option("results/tier_c_ablations.json", help="Summary JSON"),
    h6_batches: int = typer.Option(15, help="Batches for H6 quick train"),
    h6_epochs: int = typer.Option(4, help="Epochs for H6 quick train"),
    skip_h6: bool = typer.Option(False, help="Skip H6 (slow)"),
    skip_h7: bool = typer.Option(False, help="Skip H7 KL ablation"),
    skip_h5: bool = typer.Option(False, help="Skip H5 draft ablation"),
) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "ablations": {}}

    if not skip_h7:
        typer.echo("Running H7 KL ablation...")
        h7_out = out_path.parent / "kl_ablation_h7.json"
        r = _run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.ablate_kl",
                "--oracle-path",
                oracle_path,
                "--subset-limit",
                "80",
                "--epochs",
                "6",
                "--output",
                str(h7_out),
            ]
        )
        report["ablations"]["H7_kl"] = {**r, "output": str(h7_out)}
        if h7_out.exists():
            report["ablations"]["H7_kl"]["results"] = json.loads(h7_out.read_text())

    if not skip_h5:
        typer.echo("Running H5 draft-signal ablation...")
        h5_out = out_path.parent / "draft_ablation_h5.json"
        r = _run(
            [
                sys.executable,
                "-m",
                "hbac.scripts.ablate_draft",
                "--oracle-path",
                oracle_path,
                "--subset-limit",
                "80",
                "--epochs",
                "8",
                "--output",
                str(h5_out),
            ]
        )
        report["ablations"]["H5_draft"] = {**r, "output": str(h5_out)}
        if h5_out.exists():
            report["ablations"]["H5_draft"]["results"] = json.loads(h5_out.read_text())

    if not skip_h6:
        typer.echo("Running H6 counterfactual credit comparison (quick train)...")
        base = Path("checkpoints/ablations/h6")
        configs = [
            ("with_credit", ["--use-counterfactual"]),
            ("no_credit", ["--no-use-counterfactual"]),
        ]
        h6_rows = []
        for tag, flags in configs:
            out_dir = base / tag
            cmd = [
                sys.executable,
                "-m",
                "hbac.scripts.train_variant_b",
                "--oracle-path",
                oracle_path,
                "--checkpoint",
                "checkpoints/variant_a",
                "--stage",
                "3",
                "--freeze-l2",
                "--budget-fraction",
                "0.5",
                "--num-batches",
                str(h6_batches),
                "--epochs",
                str(h6_epochs),
                "--grpo-groups",
                "8",
                "--seed",
                "48",
                "--output",
                str(out_dir),
                *flags,
            ]
            r = _run(cmd)
            run_dir = sorted((out_dir / "stage3").glob("*/"), key=lambda p: p.stat().st_mtime)[-1]
            eval_out = out_path.parent / f"h6_{tag}.json"
            eval_cmd = [
                sys.executable,
                "-m",
                "hbac.scripts.eval_compose",
                "--batches-path",
                str(run_dir / "batches.jsonl"),
                "--l2-checkpoint",
                str(run_dir / "frozen_l2_controller.npz"),
                "--l1-checkpoint",
                str(run_dir / "level1_policy.npz"),
                "--oracle-path",
                oracle_path,
                "--output",
                str(eval_out),
            ]
            er = _run(eval_cmd)
            row = {"tag": tag, "train": r, "eval": er, "eval_output": str(eval_out)}
            if eval_out.exists():
                row["metrics"] = json.loads(eval_out.read_text())
            h6_rows.append(row)
        report["ablations"]["H6_counterfactual"] = h6_rows

    report["notes"] = {
        "H5_draft_signals": "featurize_observation dims 8–9; see ablate_draft.py",
        "H8_curriculum": "Requires full stage-wise retrain; deferred to Rivanna tight retrain.",
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
