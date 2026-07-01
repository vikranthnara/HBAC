from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.training.reward import TaskControllerReward
from hbac.training.validation import (
    all_passed,
    best_reward_defaults,
    run_all_validations,
    sweep_reward_hyperparameters,
)

app = typer.Typer(help="Validate Level-2 reward function anti-hacking properties (Phase 2)")


@app.command()
def main(
    lambda_token: float | None = typer.Option(None, help="Token cost coefficient λ"),
    premature_stop_penalty: float | None = typer.Option(None, help="Penalty for premature stop"),
    kl_coef: float = typer.Option(0.02, help="Reference KL coef for PPO (informational)"),
    sweep: bool = typer.Option(True, help="Run hyperparameter sweep"),
    output: str = typer.Option("results/reward_validation.json", help="JSON report path"),
) -> None:
    sweep_results = sweep_reward_hyperparameters() if sweep else []
    default_lam, default_pen = best_reward_defaults(sweep_results) if sweep else (0.001, 0.5)

    lam = lambda_token if lambda_token is not None else default_lam
    pen = premature_stop_penalty if premature_stop_penalty is not None else default_pen

    reward = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)
    results = run_all_validations(reward)
    all_ok = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        typer.echo(f"[{status}] {r.name}: {r.detail}")
        all_ok = all_ok and r.passed

    report = {
        "lambda_token": lam,
        "premature_stop_penalty": pen,
        "all_passed": all_ok,
        "validations": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
        "sweep": sweep_results,
        "recommended_defaults": {"lambda_token": default_lam, "premature_stop_penalty": default_pen},
        "kl_coef_reference": kl_coef,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"\nReport -> {out_path}")
    typer.echo(f"PPO KL config reference: kl_coef={kl_coef}")

    if not all_ok:
        typer.echo("\nReward validation FAILED — fix reward before Phase 2 training.")
        raise typer.Exit(1)
    typer.echo("\nReward validation PASSED — ready for Variant A training.")


if __name__ == "__main__":
    app()
