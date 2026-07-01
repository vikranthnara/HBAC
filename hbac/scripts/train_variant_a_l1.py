from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.batch_rollout import rollout_batch_schema
from hbac.training.controller import MonolithicController
from hbac.training.level1 import Level1Allocator
from hbac.training.reward import TaskControllerReward
from hbac.training.utility_net import UtilityNetwork

app = typer.Typer(help="Variant A Stage 3: PPO utility network Level-1 with frozen L2")


def _resolve_checkpoint(checkpoint: Path) -> Path:
    if checkpoint.is_file():
        return checkpoint
    ckpts = sorted(checkpoint.rglob("stage1_stop_controller.npz"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise typer.Exit("No L2 checkpoint found")
    return ckpts[-1]


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    checkpoint: str = typer.Option("checkpoints/variant_a", help="L2 checkpoint"),
    budget_fraction: float = typer.Option(0.90, help="B_total fraction"),
    num_batches: int = typer.Option(20, help="Training batches"),
    epochs: int = typer.Option(5, help="Training epochs"),
    learning_rate: float = typer.Option(0.01, help="Utility net LR"),
    output: str = typer.Option("checkpoints/variant_a_l1", help="Output dir"),
    seed: int = typer.Option(42, help="Seed"),
) -> None:
    np.random.seed(seed)
    root = Path(oracle_path)
    l2 = MonolithicController.load(_resolve_checkpoint(Path(checkpoint)))
    utility = UtilityNetwork()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_curriculum_batches(root, num_batches=num_batches, seed=seed)
    for b in batches:
        b.global_budget = max(len(b.tasks) * 50, int(b.oracle_token_sum * budget_fraction))
    save_batches(batches, out_dir / "batches.jsonl")

    reward_fn = TaskControllerReward()
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(epochs):
        batch_rewards: list[float] = []
        for batch in batches:
            alloc = utility.allocate_greedy(batch.tasks, batch.global_budget)
            result = rollout_batch_schema(batch, alloc, l2, reward_fn=reward_fn)
            batch_rewards.append(result.batch_reward)

            params = utility.flat_params()
            grad = np.zeros_like(params)
            delta = 1e-5
            target = result.batch_reward
            for task in batch.tasks:
                bgt = alloc[task.task_id]
                pred = utility.predict(task, bgt, batch.global_budget)
                err = pred - target
                for j in range(len(params)):
                    p_plus = params.copy()
                    p_plus[j] += delta
                    utility.load_flat_params(p_plus)
                    pred_plus = utility.predict(task, bgt, batch.global_budget)
                    grad[j] += err * (pred_plus - pred) / delta
                utility.load_flat_params(params)
            utility.load_flat_params(params + learning_rate * grad / max(len(batch.tasks), 1))

        uniform_rewards: list[float] = []
        for batch in batches:
            alloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
            uniform_rewards.append(
                rollout_batch_schema(batch, alloc, l2, reward_fn=reward_fn).batch_reward
            )

        record = {
            "epoch": epoch + 1,
            "mean_utility_reward": float(np.mean(batch_rewards)),
            "mean_uniform_reward": float(np.mean(uniform_rewards)),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        typer.echo(
            f"epoch {epoch + 1} utility={record['mean_utility_reward']:.4f} "
            f"uniform={record['mean_uniform_reward']:.4f}"
        )

    utility.save(out_dir / "utility_net.npz")
    typer.echo(f"Saved -> {out_dir}")


if __name__ == "__main__":
    app()
