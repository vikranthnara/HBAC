from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController, featurize_observation
from hbac.training.dataset import (
    find_oracle_paths,
    load_stop_examples,
    train_val_split,
)
from hbac.training.kl_calibration import tail_ref_kl_mean
from hbac.training.ppo import PPOTrainer, Transition
from hbac.training.reward import TaskControllerReward
from hbac.training.sft_warmstart import init_continue_bias, sft_warmstart_stop_head
from hbac.training.validation import all_passed, best_reward_defaults, sweep_reward_hyperparameters

app = typer.Typer(help="Train Variant A monolithic stop controller on oracle subset (Phase 2)")


def _build_batch(
    examples: list[dict],
    controller: MonolithicController,
    reward_fn: TaskControllerReward,
    budget: int = 50_000,
) -> list[Transition]:
    batch: list[Transition] = []
    for ex in examples:
        obs = ex["observation"]
        stop = ex["stop"]
        old_lp = controller.log_prob_stop(obs, stop)
        reward = reward_fn.terminal(
            success=ex["success"] and stop,
            tokens_used=ex["tokens"],
            budget=budget,
            agent_initiated_stop=stop and not ex["success"],
            env_done=stop and ex["success"],
        )
        batch.append(
            Transition(
                observation_features=featurize_observation(obs),
                stop_action=stop,
                reward=reward,
                old_log_prob=old_lp,
                done=stop,
            )
        )
    return batch


def _eval_accuracy(controller: MonolithicController, examples: list[dict]) -> float:
    if not examples:
        return 0.0
    correct = 0
    for ex in examples:
        pred = controller.should_stop(ex["observation"])
        if pred == ex["stop"]:
            correct += 1
    return correct / len(examples)


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracles dir or oracles.jsonl file"),
    subset_limit: int = typer.Option(50, help="Max training examples"),
    epochs: int = typer.Option(10, help="Training epochs"),
    kl_coef: float = typer.Option(0.02, help="Initial KL penalty coefficient"),
    val_fraction: float = typer.Option(0.2, help="Validation fraction by task"),
    seed: int = typer.Option(42, help="Random seed"),
    output: str = typer.Option("checkpoints/variant_a", help="Checkpoint root directory"),
) -> None:
    sweep = sweep_reward_hyperparameters()
    lam, pen = best_reward_defaults(sweep)
    reward_fn = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)
    if not all_passed(reward_fn):
        typer.echo("Reward validation failed. Run: python -m hbac.scripts.validate_reward")
        raise typer.Exit(1)

    root = Path(oracle_path)
    paths = find_oracle_paths(root)
    if not paths:
        typer.echo("No oracles.jsonl found. Run seed_oracles or collect_oracles.")
        raise typer.Exit(1)

    examples = load_stop_examples(paths, include_negatives=True, negative_root=root)
    if subset_limit:
        examples = examples[:subset_limit]
    if not examples:
        typer.echo("No training examples.")
        raise typer.Exit(1)

    train_ex, val_ex = train_val_split(examples, val_fraction=val_fraction, seed=seed)
    typer.echo(f"Training on {len(train_ex)} examples, val {len(val_ex)} from {len(paths)} oracle file(s)")

    np.random.seed(seed)
    controller = MonolithicController()
    init_continue_bias(controller)
    sft_losses = sft_warmstart_stop_head(controller, train_ex, epochs=150)
    typer.echo(f"SFT warm-start: final loss={sft_losses[-1]:.4f}, val_acc={_eval_accuracy(controller, val_ex):.2%}")

    ppo_cfg = PPOConfig(
        kl_coef=0.05,
        kl_adaptive=False,
        freeze_hidden=True,
        learning_rate_stop_head=5e-5,
    )
    trainer = PPOTrainer(controller, ppo_cfg, reward_fn)
    trainer.ref_controller = controller.frozen_copy()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    stats_history: list = []
    ref_kls: list[float] = []
    for epoch in range(max(epochs, 8)):
        batch = _build_batch(train_ex, controller, reward_fn)
        stats = trainer.update(batch)
        stats_history.append(stats)
        ref_kls.append(stats.ref_kl)

    for epoch, stats in enumerate(stats_history, start=1):
        val_acc = _eval_accuracy(controller, val_ex)
        record = {
            "epoch": epoch,
            "policy_loss": stats.policy_loss,
            "kl": stats.kl_divergence,
            "ref_kl": stats.ref_kl,
            "kl_coef": stats.kl_coef,
            "entropy": stats.entropy,
            "val_stop_accuracy": val_acc,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        typer.echo(
            f"epoch {epoch}/{len(stats_history)} loss={stats.policy_loss:.4f} "
            f"ref_kl={stats.ref_kl:.4f} kl_coef={stats.kl_coef:.4f} "
            f"val_acc={val_acc:.2%}"
        )

    typer.echo(f"PPO complete: tail |ref_kl|={tail_ref_kl_mean(ref_kls):.4f}")

    ckpt = out_dir / "stage1_stop_controller.npz"
    controller.save(ckpt, ref_controller=trainer.ref_controller)
    typer.echo(f"Saved checkpoint -> {ckpt}")


if __name__ == "__main__":
    app()
