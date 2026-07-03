from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from hbac.training.batch_curriculum import generate_curriculum_batches, save_batches
from hbac.training.batch_rollout import rollout_batch_schema
from hbac.training.controller import MonolithicController
from hbac.training.credit import compute_counterfactual_credits, credit_weighted_schema_reward
from hbac.training.dataset import find_oracle_paths, load_stop_examples
from hbac.training.grpo import GRPOTrainer, L2GRPOTrainer
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.phase3_pipeline import Stage3Config, evaluate_l1_policy
from hbac.training.ppo import PPOConfig, PPOTrainer
from hbac.training.reward import BatchReward, TaskControllerReward
from hbac.training.sft_warmstart import init_continue_bias, sft_warmstart_stop_head
from hbac.scripts.train_variant_a import _build_batch

app = typer.Typer(help="Variant B: hierarchical GRPO with oracle replay + counterfactual credit")


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
    checkpoint: str = typer.Option("checkpoints/variant_a", help="L2 checkpoint path or dir"),
    stage: int = typer.Option(3, help="3=frozen L2, 4=joint L2 GRPO"),
    freeze_l2: bool = typer.Option(True, help="Freeze Level-2 controller"),
    budget_fraction: float = typer.Option(0.90, help="B_total as fraction of oracle tokens"),
    grpo_groups: int = typer.Option(16, help="G allocation schemas per batch"),
    num_batches: int = typer.Option(30, help="Training batches"),
    epochs: int = typer.Option(8, help="Epochs over batch set"),
    use_counterfactual: bool = typer.Option(True, help="COMA-style L1 credit"),
    output: str = typer.Option("checkpoints/variant_b", help="Output directory"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    np.random.seed(seed)
    root = Path(oracle_path)
    ckpt_path = _resolve_checkpoint(Path(checkpoint))
    l2 = MonolithicController.load(ckpt_path)
    typer.echo(f"Loaded L2 checkpoint: {ckpt_path}")

    l1 = Level1Policy(num_schemas=min(grpo_groups, 16))
    trainer = GRPOTrainer(l1, learning_rate=0.02)
    oracle_index = OracleIndex(root)
    reward_fn = TaskControllerReward()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(output) / f"stage{stage}" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_curriculum_batches(root, num_batches=num_batches, seed=seed)
    for b in batches:
        floor = len(b.tasks) * 40
        b.global_budget = max(floor, int(b.oracle_token_sum * budget_fraction))
        b.budget_fraction = budget_fraction
    save_batches(batches, out_dir / "batches.jsonl")

    val_batches = batches[: max(1, len(batches) // 5)]
    train_batches = batches[len(val_batches) :]

    joint_l2 = stage >= 4 and not freeze_l2
    grpo_l2 = L2GRPOTrainer(l2, learning_rate=1e-4) if joint_l2 else None
    ppo = None
    if joint_l2:
        init_continue_bias(l2)
        examples = load_stop_examples(find_oracle_paths(root), limit=100)
        if examples:
            sft_warmstart_stop_head(l2, examples, epochs=50)
        ppo = PPOTrainer(l2, PPOConfig(freeze_hidden=True, learning_rate_stop_head=5e-5))
        ppo.ref_controller = l2.frozen_copy()

    log_path = out_dir / "train_log.jsonl"
    for epoch in range(epochs):
        epoch_rewards: list[float] = []
        skipped = 0
        for batch in train_batches:
            schema_ids = l1.sample_schemas(batch, grpo_groups)
            rewards: list[float] = []
            for sid in schema_ids:
                alloc = l1.allocate_schema(batch, sid)
                results = [
                    rollout_task_with_oracle(t, alloc[t.task_id], l2, oracle_index, reward_fn)
                    for t in batch.tasks
                ]
                br = l1_schema_reward(results, batch, alloc)
                if use_counterfactual:
                    rollout = rollout_batch_schema(batch, alloc, l2, schema_id=sid)
                    credits = compute_counterfactual_credits(batch, rollout, l2, oracle_index)
                    br = credit_weighted_schema_reward(credits, br, mix=0.2)
                rewards.append(br)

            stats = trainer.update_l1(batch, schema_ids, rewards)
            epoch_rewards.extend(rewards)
            if stats.skipped:
                skipped += 1

            if joint_l2 and grpo_l2 and ppo and not stats.skipped:
                ex = load_stop_examples(find_oracle_paths(root), limit=50)
                if ex:
                    grpo_l2.update_from_examples(ex)
                    ppo.update(_build_batch(ex, l2, reward_fn))

        metrics = evaluate_l1_policy(l1, val_batches, l2, oracle_index)
        record = {
            "epoch": epoch + 1,
            "mean_batch_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "gradient_starvation_skips": skipped,
            **metrics.to_dict(),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        typer.echo(
            f"epoch {epoch + 1}/{epochs} mean_R={record['mean_batch_reward']:.4f} "
            f"pass@1={metrics.pass_at_1:.2%} beats_uniform={metrics.beats_uniform}"
        )

    l1.save(out_dir / "level1_policy.npz")
    if joint_l2:
        l2.save(out_dir / "joint_l2_controller.npz")
    else:
        l2.save(out_dir / "frozen_l2_controller.npz")
    typer.echo(f"Saved -> {out_dir}")


if __name__ == "__main__":
    app()
