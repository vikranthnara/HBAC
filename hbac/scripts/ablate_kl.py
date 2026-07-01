from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController
from hbac.training.dataset import find_oracle_paths, load_stop_examples, train_val_split
from hbac.training.probes import probe_premature_stop_rate
from hbac.training.reward import TaskControllerReward
from hbac.training.validation import best_reward_defaults, sweep_reward_hyperparameters
from hbac.scripts.train_variant_a import _build_batch, _eval_accuracy

app = typer.Typer(help="KL ablation for Variant A stop controller (H7)")


def _train_with_kl(
    kl_coef: float,
    train_ex: list[dict],
    val_ex: list[dict],
    reward_fn: TaskControllerReward,
    epochs: int,
    seed: int,
) -> MonolithicController:
    from hbac.training.ppo import PPOTrainer

    np.random.seed(seed)
    controller = MonolithicController()
    trainer = PPOTrainer(controller, PPOConfig(kl_coef=kl_coef, kl_adaptive=kl_coef > 0), reward_fn)
    for _ in range(epochs):
        batch = _build_batch(train_ex, controller, reward_fn)
        trainer.update(batch)
    return controller


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracles directory"),
    subset_limit: int = typer.Option(50, help="Max examples"),
    epochs: int = typer.Option(8, help="Epochs per KL setting"),
    seed: int = typer.Option(42, help="Random seed"),
    output: str = typer.Option("results/kl_ablation.json", help="Results JSON"),
) -> None:
    kl_values = [0.0, 0.01, 0.02, 0.05]
    sweep = sweep_reward_hyperparameters()
    lam, pen = best_reward_defaults(sweep)
    reward_fn = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)

    root = Path(oracle_path)
    paths = find_oracle_paths(root)
    examples = load_stop_examples(paths, limit=subset_limit, include_negatives=True, negative_root=root)
    train_ex, val_ex = train_val_split(examples, val_fraction=0.2, seed=seed)

    results = []
    for kl in kl_values:
        controller = _train_with_kl(kl, train_ex, val_ex, reward_fn, epochs, seed)
        probe = probe_premature_stop_rate(controller)
        val_acc = _eval_accuracy(controller, val_ex)
        entry = {
            "kl_coef": kl,
            "val_stop_accuracy": val_acc,
            **probe,
        }
        results.append(entry)
        typer.echo(
            f"kl_coef={kl:.2f} val_acc={val_acc:.2%} "
            f"premature_stop_rate={probe['premature_stop_rate']:.2%} "
            f"mean_stop_prob={probe['mean_stop_prob']:.3f}"
        )

    zero_kl = next(r for r in results if r["kl_coef"] == 0.0)
    reg_kl = next(r for r in results if r["kl_coef"] == 0.02)
    h7_pass = (
        zero_kl["premature_stop_rate"] >= reg_kl["premature_stop_rate"]
        and reg_kl["val_stop_accuracy"] >= zero_kl["val_stop_accuracy"] - 0.05
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "h7_pass": h7_pass,
        "results": results,
        "criterion": "kl_coef=0 has higher premature_stop_rate than kl_coef=0.02 without large val_acc drop",
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"\nH7 {'PASS' if h7_pass else 'FAIL'} — report -> {out_path}")
    if not h7_pass:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
