"""H5 ablation: draft signals (αₜ, draft_token_frac) in L2 state features."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from hbac.training.config import PPOConfig
from hbac.training.controller import MonolithicController, attach_draft_signals, featurize_observation
from hbac.training.dataset import find_oracle_paths, load_stop_examples, train_val_split
from hbac.training.ppo import PPOTrainer, Transition
from hbac.training.reward import TaskControllerReward
from hbac.training.validation import best_reward_defaults, sweep_reward_hyperparameters
from hbac.scripts.train_variant_a import _eval_accuracy

app = typer.Typer(help="H5: draft-signal features vs 7-dim baseline (Research Plan §4)")


def _augment_examples(examples: list[dict]) -> list[dict]:
    out: list[dict] = []
    for ex in examples:
        obs = attach_draft_signals(ex["observation"])
        out.append({**ex, "observation": obs})
    return out


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
                observation_features=featurize_observation(obs, controller.input_dim),
                stop_action=stop,
                reward=reward,
                old_log_prob=old_lp,
                done=stop,
            )
        )
    return batch


def _train(
    input_dim: int,
    train_ex: list[dict],
    reward_fn: TaskControllerReward,
    epochs: int,
    seed: int,
) -> MonolithicController:
    np.random.seed(seed)
    controller = MonolithicController(input_dim=input_dim)
    trainer = PPOTrainer(controller, PPOConfig(kl_coef=0.02, kl_adaptive=True), reward_fn)
    for _ in range(epochs):
        batch = _build_batch(train_ex, controller, reward_fn)
        trainer.update(batch)
    return controller


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracles directory"),
    subset_limit: int = typer.Option(80, help="Max stop examples"),
    epochs: int = typer.Option(8, help="PPO epochs per setting"),
    seed: int = typer.Option(42, help="Random seed"),
    output: str = typer.Option("results/draft_ablation_h5.json", help="Results JSON"),
) -> None:
    sweep = sweep_reward_hyperparameters()
    lam, pen = best_reward_defaults(sweep)
    reward_fn = TaskControllerReward(lambda_token=lam, premature_stop_penalty=pen)

    root = Path(oracle_path)
    paths = find_oracle_paths(root)
    examples = load_stop_examples(paths, limit=subset_limit, include_negatives=True, negative_root=root)
    train_ex, val_ex = train_val_split(examples, val_fraction=0.2, seed=seed)
    draft_train = _augment_examples(train_ex)
    draft_val = _augment_examples(val_ex)

    baseline = _train(7, train_ex, reward_fn, epochs, seed)
    with_draft = _train(9, draft_train, reward_fn, epochs, seed)

    base_acc = _eval_accuracy(baseline, val_ex)
    draft_acc = _eval_accuracy(with_draft, draft_val)
    delta = draft_acc - base_acc

    report = {
        "hypothesis": "H5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_7dim": {"val_stop_accuracy": base_acc, "input_dim": 7},
        "with_draft_9dim": {"val_stop_accuracy": draft_acc, "input_dim": 9},
        "delta_accuracy": delta,
        "h5_supported": draft_acc > base_acc,
        "epochs": epochs,
        "subset_limit": subset_limit,
        "seed": seed,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
