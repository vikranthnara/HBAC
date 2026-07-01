from __future__ import annotations

import json
from pathlib import Path

import typer

from hbac.training.controller import MonolithicController
from hbac.training.dataset import find_oracle_paths, load_stop_examples, train_val_split
from hbac.training.probes import probe_premature_stop_rate
from hbac.training.reward import TaskControllerReward
from hbac.training.validation import best_reward_defaults, sweep_reward_hyperparameters
from hbac.scripts.train_variant_a import _eval_accuracy

app = typer.Typer(help="Evaluate Variant A stop controller checkpoint")


@app.command()
def main(
    checkpoint: str = typer.Option(..., help="Path to stage1_stop_controller.npz"),
    oracle_path: str = typer.Option("data/oracles", help="Oracles for holdout eval"),
    subset_limit: int = typer.Option(50, help="Max examples"),
    val_fraction: float = typer.Option(0.2, help="Holdout fraction"),
    seed: int = typer.Option(42, help="Split seed"),
    output: str | None = typer.Option(None, help="Optional JSON report path"),
) -> None:
    controller = MonolithicController.load(Path(checkpoint))
    sweep = sweep_reward_hyperparameters()
    lam, pen = best_reward_defaults(sweep)
    reward_fn = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)

    root = Path(oracle_path)
    paths = find_oracle_paths(root)
    examples = load_stop_examples(paths, limit=subset_limit, include_negatives=True, negative_root=root)
    _, val_ex = train_val_split(examples, val_fraction=val_fraction, seed=seed)

    val_acc = _eval_accuracy(controller, val_ex)
    probe = probe_premature_stop_rate(controller)

    tp = fp = fn = 0
    for ex in val_ex:
        pred = controller.should_stop(ex["observation"])
        label = ex["stop"]
        if pred and label:
            tp += 1
        elif pred and not label:
            fp += 1
        elif not pred and label:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    report = {
        "checkpoint": checkpoint,
        "val_stop_accuracy": val_acc,
        "precision_stop": precision,
        "recall_stop": recall,
        "probe": probe,
        "n_val_examples": len(val_ex),
    }
    typer.echo(f"Val stop accuracy: {val_acc:.2%}")
    typer.echo(f"Precision (stop): {precision:.2%}  Recall (stop): {recall:.2%}")
    typer.echo(
        f"Probe premature_stop_rate: {probe['premature_stop_rate']:.2%} "
        f"mean_stop_prob: {probe['mean_stop_prob']:.3f}"
    )

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(f"Report -> {out_path}")


if __name__ == "__main__":
    app()
