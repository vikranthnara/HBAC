"""Full Phase 3 training pipeline: Stages 3→4 with evaluation gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hbac.training.batch_curriculum import TrainingBatch, generate_curriculum_batches, save_batches
from hbac.training.batch_rollout import rollout_batch_schema
from hbac.training.controller import MonolithicController
from hbac.training.credit import compute_counterfactual_credits, credit_weighted_schema_reward
from hbac.training.dataset import find_oracle_paths, load_stop_examples
from hbac.training.grpo import GRPOTrainer
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.oracle_replay import OracleIndex, rollout_task_with_oracle
from hbac.training.l1_batch_reward import (
    batch_violation_count,
    domain_allocation_variance,
    l1_schema_reward,
)
from hbac.training.reward import BatchReward, TaskControllerReward
from hbac.training.sft_warmstart import init_continue_bias, sft_warmstart_stop_head
from hbac.training.utility_net import UtilityNetwork
from hbac.scripts.train_variant_a import _build_batch
from hbac.training.ppo import PPOConfig, PPOTrainer


@dataclass
class Stage3Config:
    grpo_groups: int = 16
    num_batches: int = 30
    epochs: int = 8
    budget_fraction: float = 0.70
    learning_rate: float = 0.02
    use_counterfactual: bool = True
    seed: int = 42


@dataclass
class EvalMetrics:
    pass_at_1: float
    uniform_pass_at_1: float
    batch_violation_rate: float
    mean_batch_reward: float
    uniform_mean_reward: float
    beats_uniform: bool
    allocation_variance: float
    domain_allocation_variance: float

    def to_dict(self) -> dict:
        return {
            "pass_at_1": self.pass_at_1,
            "uniform_pass_at_1": self.uniform_pass_at_1,
            "batch_violation_rate": self.batch_violation_rate,
            "mean_batch_reward": self.mean_batch_reward,
            "uniform_mean_reward": self.uniform_mean_reward,
            "beats_uniform": self.beats_uniform,
            "allocation_variance": self.allocation_variance,
            "domain_allocation_variance": self.domain_allocation_variance,
        }


@dataclass
class Phase3Result:
    stage3_dir: Path
    stage4_dir: Path | None
    variant_a_dir: Path | None
    stage3_metrics: EvalMetrics
    stage4_metrics: EvalMetrics | None
    logs: list[dict] = field(default_factory=list)


def _resolve_l2(checkpoint: Path) -> MonolithicController:
    if checkpoint.is_file():
        return MonolithicController.load(checkpoint)
    ckpts = sorted(checkpoint.rglob("stage1_stop_controller.npz"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No L2 checkpoint under {checkpoint}")
    return MonolithicController.load(ckpts[-1])


def evaluate_l1_policy(
    l1: Level1Policy,
    batches: list[TrainingBatch],
    l2: MonolithicController,
    oracle_index: OracleIndex,
) -> EvalMetrics:
    batch_reward_fn = BatchReward()
    policy_rewards: list[float] = []
    uniform_rewards: list[float] = []
    policy_successes: list[bool] = []
    uniform_successes: list[bool] = []
    violations = 0
    alloc_vars: list[float] = []
    domain_vars: list[float] = []

    for batch in batches:
        sid = int(np.argmax(l1.schema_probs(batch)))
        alloc = l1.allocate_schema(batch, sid)
        alloc_vars.append(float(np.var(list(alloc.values()))) if alloc else 0.0)
        domain_vars.append(domain_allocation_variance(alloc, batch))

        results = []
        for task in batch.tasks:
            r = rollout_task_with_oracle(
                task, alloc[task.task_id], l2, oracle_index
            )
            results.append(r)
            policy_successes.append(r.success)
            if r.budget_violated:
                violations += 1

        policy_rewards.append(l1_schema_reward(results, batch, alloc))

        ualloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        uresults = [
            rollout_task_with_oracle(task, ualloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        uniform_successes.extend(r.success for r in uresults)
        uniform_rewards.append(l1_schema_reward(uresults, batch, ualloc))
        _, batch_viol = batch_violation_count(results, global_budget=batch.global_budget)
        violations += batch_viol

    mean_p = float(np.mean(policy_rewards)) if policy_rewards else 0.0
    mean_u = float(np.mean(uniform_rewards)) if uniform_rewards else 0.0
    pass_p = sum(policy_successes) / max(len(policy_successes), 1)
    pass_u = sum(uniform_successes) / max(len(uniform_successes), 1)
    from hbac.gates.phase3_thresholds import PHASE3A

    return EvalMetrics(
        pass_at_1=pass_p,
        uniform_pass_at_1=pass_u,
        batch_violation_rate=violations / max(len(policy_successes) + len(batches), 1),
        mean_batch_reward=mean_p,
        uniform_mean_reward=mean_u,
        beats_uniform=pass_p > pass_u + PHASE3A.min_pass_at_1_margin,
        allocation_variance=float(np.mean(alloc_vars)) if alloc_vars else 0.0,
        domain_allocation_variance=float(np.mean(domain_vars)) if domain_vars else 0.0,
    )


def train_stage3_variant_b(
    oracle_root: Path,
    l2_ckpt: Path,
    out_dir: Path,
    cfg: Stage3Config,
) -> tuple[Level1Policy, list[TrainingBatch], EvalMetrics]:
    np.random.seed(cfg.seed)
    l2 = _resolve_l2(l2_ckpt)
    l1 = Level1Policy(num_schemas=min(cfg.grpo_groups, 16))
    trainer = GRPOTrainer(l1, learning_rate=cfg.learning_rate)
    oracle_index = OracleIndex(oracle_root)
    reward_fn = TaskControllerReward()

    batches = generate_curriculum_batches(oracle_root, num_batches=cfg.num_batches, seed=cfg.seed)
    for b in batches:
        b.global_budget = max(len(b.tasks) * 40, int(b.oracle_token_sum * cfg.budget_fraction))
        b.budget_fraction = cfg.budget_fraction
    save_batches(batches, out_dir / "batches.jsonl")

    val_batches = batches[: max(1, len(batches) // 5)]
    train_batches = batches[len(val_batches) :]
    eval_batches = generate_curriculum_batches(oracle_root, num_batches=10, seed=cfg.seed + 999)
    for b in eval_batches:
        b.global_budget = max(len(b.tasks) * 40, int(b.oracle_token_sum * b.budget_fraction))

    log_path = out_dir / "train_log.jsonl"
    total_batches = 0
    for epoch in range(cfg.epochs):
        skipped = 0
        epoch_rewards: list[float] = []
        for batch in train_batches:
            total_batches += 1
            schema_ids = l1.sample_schemas(batch, cfg.grpo_groups)
            rewards: list[float] = []
            for sid in schema_ids:
                alloc = l1.allocate_schema(batch, sid)
                results = [
                    rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index, reward_fn)
                    for task in batch.tasks
                ]
                br = l1_schema_reward(results, batch, alloc)
                if cfg.use_counterfactual:
                    from hbac.training.batch_rollout import BatchRolloutResult

                    rollout_result = BatchRolloutResult(
                        schema_id=sid,
                        allocations=alloc,
                        task_results=results,
                        batch_reward=br,
                    )
                    credits = compute_counterfactual_credits(
                        batch, rollout_result, l2, oracle_index, cached_results=results
                    )
                    br = credit_weighted_schema_reward(credits, br, mix=0.2)
                rewards.append(br)

            stats = trainer.update_l1(batch, schema_ids, rewards)
            epoch_rewards.extend(rewards)
            if stats.skipped:
                skipped += 1

        metrics = evaluate_l1_policy(l1, val_batches, l2, oracle_index)
        record = {
            "epoch": epoch + 1,
            "mean_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "gradient_starvation_skips": skipped,
            "gradient_starvation_rate": skipped / max(len(train_batches), 1),
            **metrics.to_dict(),
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    l1.save(out_dir / "level1_policy.npz")
    l2.save(out_dir / "frozen_l2_controller.npz")
    final_metrics = evaluate_l1_policy(l1, eval_batches, l2, oracle_index)
    return l1, batches, final_metrics


def train_stage4_joint(
    oracle_root: Path,
    l1: Level1Policy,
    l2_ckpt: Path,
    out_dir: Path,
    cfg: Stage3Config,
) -> tuple[MonolithicController, EvalMetrics]:
    np.random.seed(cfg.seed)
    l2 = _resolve_l2(l2_ckpt)
    init_continue_bias(l2)
    examples = load_stop_examples(find_oracle_paths(oracle_root), limit=200)
    if examples:
        sft_warmstart_stop_head(l2, examples, epochs=80)

    ppo = PPOTrainer(l2, PPOConfig(freeze_hidden=True, learning_rate_stop_head=5e-5))
    ppo.ref_controller = l2.frozen_copy()

    from hbac.training.grpo import L2GRPOTrainer

    grpo_l2 = L2GRPOTrainer(l2, learning_rate=1e-4)

    oracle_index = OracleIndex(oracle_root)
    batches = generate_curriculum_batches(oracle_root, num_batches=cfg.num_batches, seed=cfg.seed + 1)
    for b in batches:
        b.global_budget = max(len(b.tasks) * 40, int(b.oracle_token_sum * 0.75))
        b.budget_fraction = 0.75
    l1_trainer = GRPOTrainer(l1, learning_rate=cfg.learning_rate * 0.5)

    for epoch in range(max(3, cfg.epochs // 2)):
        for batch in batches:
            schema_ids = l1.sample_schemas(batch, min(cfg.grpo_groups, 8))
            rewards = []
            for sid in schema_ids:
                alloc = l1.allocate_schema(batch, sid)
                results = [
                    rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
                    for task in batch.tasks
                ]
                rewards.append(l1_schema_reward(results, batch, alloc))
            l1_trainer.update_l1(batch, schema_ids, rewards)

            ex = load_stop_examples(find_oracle_paths(oracle_root), limit=50)
            if ex:
                grpo_l2.update_from_examples(ex)
                ppo.update(_build_batch(ex, l2, TaskControllerReward()))

    l1.save(out_dir / "level1_policy.npz")
    l2.save(out_dir / "joint_l2_controller.npz")
    metrics = evaluate_l1_policy(l1, batches[:5], l2, oracle_index)
    return l2, metrics


def train_variant_a_l1_full(
    oracle_root: Path,
    l2_ckpt: Path,
    out_dir: Path,
    cfg: Stage3Config,
) -> tuple[UtilityNetwork, EvalMetrics]:
    np.random.seed(cfg.seed)
    l2 = _resolve_l2(l2_ckpt)
    utility = UtilityNetwork()
    oracle_index = OracleIndex(oracle_root)
    batches = generate_curriculum_batches(oracle_root, num_batches=cfg.num_batches, seed=cfg.seed)
    reward_fn = TaskControllerReward()
    lr = 0.01

    for epoch in range(cfg.epochs):
        for batch in batches:
            alloc = utility.allocate_greedy(batch.tasks, batch.global_budget)
            results = [
                rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index, reward_fn)
                for task in batch.tasks
            ]
            target = BatchReward().total(
                successes=[r.success for r in results],
                tokens=[r.tokens_used for r in results],
                budgets=[r.budget for r in results],
                global_budget=batch.global_budget,
            )
            params = utility.flat_params()
            grad = np.zeros_like(params)
            delta = 1e-5
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
            utility.load_flat_params(params + lr * grad / max(len(batch.tasks), 1))

    utility.save(out_dir / "utility_net.npz")
    metrics = evaluate_utility_allocation(utility, batches[:5], l2, oracle_index)
    return utility, metrics


def evaluate_utility_allocation(
    utility: UtilityNetwork,
    batches: list[TrainingBatch],
    l2: MonolithicController,
    oracle_index: OracleIndex,
) -> EvalMetrics:
    batch_reward_fn = BatchReward()
    policy_rewards: list[float] = []
    uniform_rewards: list[float] = []
    successes: list[bool] = []
    violations = 0

    for batch in batches:
        alloc = utility.allocate_greedy(batch.tasks, batch.global_budget)
        results = [
            rollout_task_with_oracle(task, alloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        successes.extend(r.success for r in results)
        policy_rewards.append(
            batch_reward_fn.total(
                successes=[r.success for r in results],
                tokens=[r.tokens_used for r in results],
                budgets=[r.budget for r in results],
                global_budget=batch.global_budget,
            )
        )
        ualloc = Level1Allocator(batch.global_budget).allocate(batch.task_ids)
        uresults = [
            rollout_task_with_oracle(task, ualloc[task.task_id], l2, oracle_index)
            for task in batch.tasks
        ]
        uniform_rewards.append(
            batch_reward_fn.total(
                successes=[r.success for r in uresults],
                tokens=[r.tokens_used for r in uresults],
                budgets=[r.budget for r in uresults],
                global_budget=batch.global_budget,
            )
        )
        violations += sum(1 for r in results if r.tokens_used > r.budget)

    mean_p = float(np.mean(policy_rewards)) if policy_rewards else 0.0
    mean_u = float(np.mean(uniform_rewards)) if uniform_rewards else 0.0
    n = max(len(successes), 1)
    return EvalMetrics(
        pass_at_1=sum(successes) / n,
        uniform_pass_at_1=0.0,
        batch_violation_rate=violations / (n + len(batches)),
        mean_batch_reward=mean_p,
        uniform_mean_reward=mean_u,
        beats_uniform=mean_p >= mean_u,
        allocation_variance=0.0,
        domain_allocation_variance=0.0,
    )


def run_full_phase3(
    oracle_root: Path,
    l2_ckpt: Path,
    output: Path,
    cfg: Stage3Config | None = None,
    *,
    run_stage4: bool = True,
    run_variant_a: bool = True,
) -> Phase3Result:
    cfg = cfg or Stage3Config()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output / run_id
    stage3_dir = base / "stage3"
    stage3_dir.mkdir(parents=True, exist_ok=True)

    _, _, stage3_metrics = train_stage3_variant_b(oracle_root, l2_ckpt, stage3_dir, cfg)

    stage4_dir = None
    stage4_metrics = None
    from hbac.gates.phase3_thresholds import PHASE3A

    stage3_ok = (
        stage3_metrics.beats_uniform
        and stage3_metrics.batch_violation_rate <= PHASE3A.max_batch_violation_rate
        and stage3_metrics.pass_at_1 >= PHASE3A.min_pass_at_1_floor
    )
    if run_stage4 and stage3_ok:
        stage4_dir = base / "stage4"
        stage4_dir.mkdir(parents=True, exist_ok=True)
        l1 = Level1Policy.load(stage3_dir / "level1_policy.npz")
        _, stage4_metrics = train_stage4_joint(
            oracle_root, l1, stage3_dir / "frozen_l2_controller.npz", stage4_dir, cfg
        )

    variant_a_dir = None
    variant_a_metrics = None
    if run_variant_a:
        variant_a_dir = base / "variant_a_l1"
        variant_a_dir.mkdir(parents=True, exist_ok=True)
        _, variant_a_metrics = train_variant_a_l1_full(oracle_root, l2_ckpt, variant_a_dir, cfg)

    report = Phase3Result(
        stage3_dir=stage3_dir,
        stage4_dir=stage4_dir,
        variant_a_dir=variant_a_dir,
        stage3_metrics=stage3_metrics,
        stage4_metrics=stage4_metrics,
    )
    (base / "phase3_report.json").write_text(
        json.dumps(
            {
                "stage3": stage3_metrics.to_dict(),
                "stage4": stage4_metrics.to_dict() if stage4_metrics else None,
                "stage4_ran": stage4_dir is not None,
                "variant_a": variant_a_metrics.to_dict() if variant_a_metrics else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
