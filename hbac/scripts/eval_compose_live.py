"""Compose-vs-joint evaluation with live LLM rollouts (not oracle replay)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator, allocation_variance
from hbac.baselines.clear_official import CLEAROfficialAllocator
from hbac.baselines.heuristics import (
    BatchReFORCProxyAllocator,
    BatchTABProxyAllocator,
    SJFAllocator,
    TypePriorAllocator,
)
from hbac.baselines.reforc_official import ReFORCOfficialAllocator
from hbac.baselines.zebra import ZEBRAAllocator
from hbac.baselines.zebra_official import ZEBRAOfficialAllocator
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend
from hbac.dotenv_loader import load_project_env
from hbac.training.batch_curriculum import TrainingBatch, load_batches
from hbac.training.batch_rollout import rollout_task
from hbac.training.controller import MonolithicController
from hbac.training.l1_batch_reward import l1_schema_reward
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.phase3_pipeline import _resolve_l2
from hbac.training.scarcity import fairness_reserve_alloc, roi_skip_alloc, scarcity_boost_alloc
from hbac.training.reward import TaskControllerReward

app = typer.Typer(help="Live-LLM compose eval: uniform vs CLEAR vs HBAC")

STUB_BENCHMARKS = frozenset({"tau_bench", "toolbench", "mock", "swe_bench", "livecodebench"})
LIVE_MIN_PER_TASK = 600
ALLOCATOR_KEYS = (
    "uniform",
    "clear_compose",
    "zebra_compose",
    "sjf_compose",
    "type_prior",
    "tab_proxy",
    "reforc_proxy",
    "clear_official",
    "zebra_official",
    "reforc_official",
    "hbac_joint",
    "hbac_d18",
    "hbac_guardrail",
    "hbac_fair",  # alias of hbac_guardrail (deprecated)
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_shard(checkpoint_dir: Path, key: str, payload: dict) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{key}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _bootstrap_ci(successes: list[bool], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not successes:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(successes, dtype=float)
    samples = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _filter_batches(
    batches: list[TrainingBatch],
    *,
    benchmarks: set[str] | None,
    max_batches: int | None,
    live: bool = True,
    live_min_per_task: int = LIVE_MIN_PER_TASK,
    budget_fraction_override: float | None = None,
) -> list[TrainingBatch]:
    out: list[TrainingBatch] = []
    for batch in batches:
        tasks = [t for t in batch.tasks if not benchmarks or t.benchmark in benchmarks]
        if not tasks:
            continue
        oracle_sum = sum(t.oracle_tokens for t in tasks) or 1
        n = len(tasks)
        frac = budget_fraction_override if budget_fraction_override is not None else batch.budget_fraction
        frac_budget = int(oracle_sum * frac)
        full_oracle = batch.oracle_token_sum or 1
        scaled_train_budget = int(batch.global_budget * (oracle_sum / full_oracle))
        if budget_fraction_override is not None and full_oracle:
            scaled_train_budget = int(oracle_sum * frac)
        floor = n * (live_min_per_task if live else 40)
        out.append(
            TrainingBatch(
                batch_id=batch.batch_id,
                tasks=tasks,
                global_budget=max(floor, scaled_train_budget, frac_budget),
                oracle_token_sum=oracle_sum,
                budget_fraction=frac,
            )
        )
        if max_batches and len(out) >= max_batches:
            break
    return out


def _eval_allocator(
    name: str,
    batches: list[TrainingBatch],
    l2: MonolithicController,
    llm: LLMBackend,
    alloc_fn,
    *,
    use_controller_stop: bool = False,
    save_per_task: bool = False,
) -> dict:
    reward_fn = TaskControllerReward()
    rewards: list[float] = []
    successes: list[bool] = []
    violations = 0
    alloc_vars: list[float] = []
    tokens_used: list[int] = []
    parse_failures: list[int] = []
    first_step_valid: list[bool] = []
    per_task_ids: list[str] = []
    by_benchmark: dict[str, dict[str, list]] = {}
    total_tasks = sum(len(b.tasks) for b in batches)
    done = 0

    for batch_idx, batch in enumerate(batches):
        alloc = alloc_fn(batch)
        alloc_vars.append(allocation_variance(alloc))
        results = []
        for task in batch.tasks:
            r = rollout_task(
                task,
                alloc[task.task_id],
                l2,
                reward_fn,
                llm=llm,
                use_controller_stop=use_controller_stop,
            )
            results.append(r)
            done += 1
            if done == 1 or done % 25 == 0 or done == total_tasks:
                _log(f"[{name}] batch {batch_idx + 1}/{len(batches)} task {done}/{total_tasks}")
            successes.append(r.success)
            if save_per_task:
                per_task_ids.append(task.task_id)
            tokens_used.append(r.tokens_used)
            parse_failures.append(r.parse_failures)
            first_step_valid.append(r.first_step_valid_json)
            bench = by_benchmark.setdefault(
                r.benchmark,
                {"successes": [], "parse_failures": [], "first_step_valid": [], "n": 0},
            )
            bench["successes"].append(r.success)
            bench["parse_failures"].append(r.parse_failures)
            bench["first_step_valid"].append(r.first_step_valid_json)
            bench["n"] += 1
            if r.budget_violated:
                violations += 1
        rewards.append(l1_schema_reward(results, batch, alloc))
        if sum(r.tokens_used for r in results) > batch.global_budget:
            violations += 1

    n = max(len(successes), 1)
    per_benchmark = {}
    for bench, stats in sorted(by_benchmark.items()):
        bn = max(stats["n"], 1)
        total_parse = sum(stats["parse_failures"])
        total_steps = max(total_parse, bn)  # lower bound; at least one step per task
        per_benchmark[bench] = {
            "pass_at_1": sum(stats["successes"]) / bn,
            "num_tasks": stats["n"],
            "mean_parse_failures_per_task": total_parse / bn,
            "first_step_valid_json_rate": sum(stats["first_step_valid"]) / bn,
        }

    out = {
        "allocator": name,
        "pass_at_1": sum(successes) / n,
        "pass_at_1_ci95": list(_bootstrap_ci(successes)),
        "mean_batch_reward": float(np.mean(rewards)) if rewards else 0.0,
        "batch_violation_rate": violations / (n + len(batches)),
        "mean_allocation_variance": float(np.mean(alloc_vars)) if alloc_vars else 0.0,
        "mean_tokens_used": float(np.mean(tokens_used)) if tokens_used else 0.0,
        "mean_parse_failures_per_task": float(np.mean(parse_failures)) if parse_failures else 0.0,
        "first_step_valid_json_rate": sum(first_step_valid) / n,
        "num_tasks": len(successes),
        "num_batches": len(batches),
        "per_benchmark": per_benchmark,
    }
    if save_per_task:
        out["per_task_ids"] = per_task_ids
        out["per_task_successes"] = successes
    return out


def _build_allocator_registry(
    *,
    l1: Level1Policy,
    l1_d18: Level1Policy | None,
    clear: CLEARAllocator,
    zebra: ZEBRAAllocator,
    clear_official: CLEAROfficialAllocator,
    zebra_official: ZEBRAOfficialAllocator,
    reforc_official: ReFORCOfficialAllocator,
    live_min_per_task: int,
    scarcity_boost: bool,
    shift_fraction: float,
    swe_min_reserve: float,
    roi_skip: bool,
    fairness_reserve: bool,
    hard_min_frac: float,
) -> dict[str, Callable]:
    def _hbac_alloc(b: TrainingBatch) -> dict[str, int]:
        alloc = l1.allocate_schema(b, int(np.argmax(l1.schema_probs(b))))
        if scarcity_boost:
            alloc = scarcity_boost_alloc(
                alloc, b, shift_fraction=shift_fraction, swe_min_reserve=swe_min_reserve
            )
        if roi_skip:
            alloc = roi_skip_alloc(alloc, b, floor_threshold=live_min_per_task)
        if fairness_reserve:
            alloc = fairness_reserve_alloc(alloc, b, hard_min_frac=hard_min_frac)
        return alloc

    def _hbac_guardrail_alloc(b: TrainingBatch) -> dict[str, int]:
        alloc = l1.allocate_schema(b, int(np.argmax(l1.schema_probs(b))))
        return fairness_reserve_alloc(alloc, b, hard_min_frac=hard_min_frac)

    def _hbac_d18_alloc(b: TrainingBatch) -> dict[str, int]:
        policy = l1_d18 if l1_d18 is not None else l1
        return policy.allocate_schema(b, int(np.argmax(policy.schema_probs(b))))

    reg = {
        "uniform": lambda b: Level1Allocator(b.global_budget).allocate(b.task_ids),
        "clear_compose": lambda b: clear.allocate(b.tasks, b.global_budget),
        "zebra_compose": lambda b: zebra.allocate(b.tasks, b.global_budget),
        "sjf_compose": lambda b: SJFAllocator(min_per_task=1).allocate(b.tasks, b.global_budget),
        "type_prior": lambda b: TypePriorAllocator().allocate(b.tasks, b.global_budget),
        "tab_proxy": lambda b: BatchTABProxyAllocator(min_per_task=live_min_per_task).allocate(
            b.tasks, b.global_budget
        ),
        "reforc_proxy": lambda b: BatchReFORCProxyAllocator(min_per_task=live_min_per_task).allocate(
            b.tasks, b.global_budget
        ),
        "clear_official": lambda b: clear_official.allocate(b.tasks, b.global_budget),
        "zebra_official": lambda b: zebra_official.allocate(b.tasks, b.global_budget),
        "reforc_official": lambda b: reforc_official.allocate(b.tasks, b.global_budget),
        "hbac_joint": _hbac_alloc,
        "hbac_d18": _hbac_d18_alloc,
        "hbac_guardrail": _hbac_guardrail_alloc,
        "hbac_fair": _hbac_guardrail_alloc,
    }
    return reg


def _run_allocators(
    keys: list[str],
    batches: list[TrainingBatch],
    l2: MonolithicController,
    llm: LLMBackend,
    registry: dict[str, Callable],
    *,
    use_controller_stop: bool,
    checkpoint_dir: Path | None,
    save_per_task: bool = False,
) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for key in keys:
        if checkpoint_dir:
            shard = checkpoint_dir / f"{key}.json"
            if shard.is_file():
                _log(f"[skip] {key} shard exists: {shard}")
                rows[key] = json.loads(shard.read_text(encoding="utf-8"))["result"]
                continue
        _log(f"[start] allocator={key}")
        row = _eval_allocator(
            key,
            batches,
            l2,
            llm,
            registry[key],
            use_controller_stop=use_controller_stop,
            save_per_task=save_per_task,
        )
        rows[key] = row
        if checkpoint_dir:
            shard_path = _write_shard(
                checkpoint_dir,
                key,
                {"allocator": key, "result": row},
            )
            _log(f"[done] {key} -> {shard_path}")
        else:
            _log(f"[done] {key} pass@1={row['pass_at_1']:.4f}")
    return rows


def _build_report(
    rows: dict[str, dict],
    *,
    llm: LLMBackend,
    lora_path: str | None,
    runner: str,
    scarcity_boost: bool,
    shift_fraction: float,
    swe_min_reserve: float,
    roi_skip: bool,
    fairness_reserve: bool,
    hard_min_frac: float,
    batches: list[TrainingBatch],
    bench_set: set[str],
    budget_fraction: float | None,
    live_min_per_task: int,
) -> dict:
    hbac = rows["hbac_joint"]
    hbac_d18 = rows.get("hbac_d18", hbac)
    hbac_guardrail = rows.get("hbac_guardrail", rows.get("hbac_fair", hbac))
    hbac_fair = hbac_guardrail
    uniform = rows["uniform"]
    clear_compose = rows["clear_compose"]
    zebra_compose = rows["zebra_compose"]
    type_prior = rows["type_prior"]
    return {
        "llm": f"{llm.config.provider}:{llm.config.model}",
        "lora_path": lora_path,
        "runner": runner,
        "scarcity_boost": scarcity_boost,
        "shift_fraction": shift_fraction if scarcity_boost else None,
        "swe_min_reserve": swe_min_reserve if scarcity_boost else None,
        "roi_skip": roi_skip,
        "fairness_reserve": fairness_reserve,
        "hard_min_frac": hard_min_frac if fairness_reserve else None,
        "num_tasks": sum(len(b.tasks) for b in batches),
        "benchmarks": sorted(bench_set),
        "budget_fraction": budget_fraction
        if budget_fraction is not None
        else (batches[0].budget_fraction if batches else None),
        "uniform": uniform,
        "clear_compose": clear_compose,
        "zebra_compose": zebra_compose,
        "sjf_compose": rows["sjf_compose"],
        "type_prior": type_prior,
        "tab_proxy": rows["tab_proxy"],
        "reforc_proxy": rows["reforc_proxy"],
        "clear_official": rows["clear_official"],
        "zebra_official": rows["zebra_official"],
        "reforc_official": rows["reforc_official"],
        "hbac_joint": hbac,
        "hbac_d18": hbac_d18,
        "hbac_guardrail": hbac_guardrail,
        "hbac_fair": hbac_fair,
        "live_min_per_task": live_min_per_task,
        "hbac_beats_clear": hbac["pass_at_1"] > clear_compose["pass_at_1"]
        or hbac["mean_batch_reward"] > clear_compose["mean_batch_reward"],
        "hbac_beats_uniform": hbac["pass_at_1"] > uniform["pass_at_1"]
        or hbac["mean_batch_reward"] > uniform["mean_batch_reward"],
        "hbac_beats_zebra": hbac["pass_at_1"] > zebra_compose["pass_at_1"]
        or hbac["mean_batch_reward"] > zebra_compose["mean_batch_reward"],
        "hbac_d18_beats_type_prior": hbac_d18["pass_at_1"] > type_prior["pass_at_1"],
        "hbac_guardrail_beats_type_prior": hbac_guardrail["pass_at_1"] > type_prior["pass_at_1"],
        "hbac_fair_beats_type_prior": hbac_guardrail["pass_at_1"] > type_prior["pass_at_1"],
        "proxy_disclaimer": (
            "clear_official/zebra_official/reforc_official: Tier-A paper algorithms. "
            "Heuristics are reviewer baselines."
        ),
    }


@app.command()
def main(
    batches_path: str = typer.Option(..., help="batches.jsonl from training run"),
    l2_checkpoint: str = typer.Option(..., help="Frozen L2 checkpoint"),
    l1_checkpoint: str = typer.Option(..., help="Learned HBAC L1 .npz"),
    output: str = typer.Option("results/compose_live.json", help="Metrics output"),
    llm_spec: str = typer.Option("auto", help="LLM spec: auto | provider:model"),
    benchmarks: str = typer.Option(
        "tau_bench,toolbench,mock,swe_bench",
        help="Comma-separated benchmarks (live eval; LCB needs oracle replay)",
    ),
    max_batches: int = typer.Option(50, help="Cap batches for API cost control (0 = all)"),
    live_min_per_task: int = typer.Option(
        LIVE_MIN_PER_TASK, help="Minimum per-task token floor for live rollouts"
    ),
    budget_fraction: float | None = typer.Option(
        None, help="Override batch budget fraction (e.g. 0.25 for scarcity sweep)"
    ),
    lora_path: str | None = typer.Option(None, help="PEFT LoRA adapter directory (Phase 3b)"),
    runner: str = typer.Option(
        "react",
        help="Step loop: react (default live) or controller (L2 stop head, Discovery D6)",
    ),
    scarcity_boost: bool = typer.Option(
        False,
        help="Inference-time budget shift from SWE→tool tasks under tight caps (D12)",
    ),
    shift_fraction: float = typer.Option(
        0.15, help="Fraction of mean alloc to shift per SWE donor (D12)"
    ),
    swe_min_reserve: float = typer.Option(
        0.5, help="Min fraction of donor budget to keep on SWE (D12 guard)"
    ),
    roi_skip: bool = typer.Option(
        False,
        help="Skip hard benchmarks (SWE/LCB) budget when floor<350; redistribute to tools (D14)",
    ),
    fairness_reserve: bool = typer.Option(
        False,
        help="Apply fairness_reserve_alloc to HBAC (D17: beat type-prior without starvation)",
    ),
    hard_min_frac: float = typer.Option(0.15, help="Min hard-benchmark fraction of uniform share"),
    only_allocator: str | None = typer.Option(
        None,
        help="Run a single allocator key (for Slurm array shards)",
    ),
    checkpoint_dir: str | None = typer.Option(
        None,
        help="Write per-allocator JSON shards (resume-safe)",
    ),
    l1_checkpoint_d18: str | None = typer.Option(
        None,
        help="D18 fairness-trained L1 for hbac_d18 (no inference guardrail)",
    ),
    save_per_task: bool = typer.Option(
        False,
        help="Store per_task_ids and per_task_successes for paired McNemar",
    ),
) -> None:
    load_project_env()
    bench_set = {b.strip() for b in benchmarks.split(",") if b.strip()}
    cap = max_batches if max_batches > 0 else None
    batches = _filter_batches(
        load_batches(Path(batches_path)),
        benchmarks=bench_set,
        max_batches=cap,
        live_min_per_task=live_min_per_task,
        budget_fraction_override=budget_fraction,
    )
    if not batches:
        raise typer.BadParameter(f"No batches with benchmarks {bench_set}")

    l2 = _resolve_l2(Path(l2_checkpoint))
    l1 = Level1Policy.load(Path(l1_checkpoint))
    l1_d18 = Level1Policy.load(Path(l1_checkpoint_d18)) if l1_checkpoint_d18 else None
    if llm_spec == "auto":
        cfg = LLMConfig()
        if lora_path:
            cfg = cfg.model_copy(update={"lora_path": lora_path})
        llm = LLMBackend.from_config(cfg)
    else:
        llm = LLMBackend.from_spec(llm_spec, lora_path=lora_path)
    clear = CLEARAllocator()
    zebra = ZEBRAAllocator()
    clear_official = CLEAROfficialAllocator(min_per_task=live_min_per_task)
    zebra_official = ZEBRAOfficialAllocator(min_per_task=live_min_per_task)
    reforc_official = ReFORCOfficialAllocator(min_per_task=live_min_per_task)

    typer.echo(f"Live LLM: {llm.config.provider}:{llm.config.model}")
    typer.echo(f"Batches: {len(batches)} tasks: {sum(len(b.tasks) for b in batches)}")

    if runner not in {"react", "controller"}:
        raise typer.BadParameter("runner must be react or controller")
    use_controller_stop = runner == "controller"

    registry = _build_allocator_registry(
        l1=l1,
        l1_d18=l1_d18,
        clear=clear,
        zebra=zebra,
        clear_official=clear_official,
        zebra_official=zebra_official,
        reforc_official=reforc_official,
        live_min_per_task=live_min_per_task,
        scarcity_boost=scarcity_boost,
        shift_fraction=shift_fraction,
        swe_min_reserve=swe_min_reserve,
        roi_skip=roi_skip,
        fairness_reserve=fairness_reserve,
        hard_min_frac=hard_min_frac,
    )

    ckpt = Path(checkpoint_dir) if checkpoint_dir else None
    if only_allocator:
        if only_allocator not in registry:
            raise typer.BadParameter(
                f"Unknown allocator {only_allocator!r}; choose from {', '.join(ALLOCATOR_KEYS)}"
            )
        keys = [only_allocator]
    else:
        keys = list(ALLOCATOR_KEYS)

    if ckpt:
        meta = {
            "llm": f"{llm.config.provider}:{llm.config.model}",
            "lora_path": lora_path,
            "runner": runner,
            "scarcity_boost": scarcity_boost,
            "shift_fraction": shift_fraction if scarcity_boost else None,
            "swe_min_reserve": swe_min_reserve if scarcity_boost else None,
            "roi_skip": roi_skip,
            "fairness_reserve": fairness_reserve,
            "hard_min_frac": hard_min_frac if fairness_reserve else None,
            "num_tasks": sum(len(b.tasks) for b in batches),
            "benchmarks": sorted(bench_set),
            "budget_fraction": budget_fraction
            if budget_fraction is not None
            else (batches[0].budget_fraction if batches else None),
            "live_min_per_task": live_min_per_task,
            "batches_path": batches_path,
        }
        _write_shard(ckpt, "meta", meta)

    rows = _run_allocators(
        keys,
        batches,
        l2,
        llm,
        registry,
        use_controller_stop=use_controller_stop,
        checkpoint_dir=ckpt,
        save_per_task=save_per_task,
    )

    if only_allocator:
        row = rows[only_allocator]
        _log(
            f"Shard complete: {only_allocator} pass@1={row['pass_at_1']:.4f} "
            f"tasks={row['num_tasks']}"
        )
        if output:
            # Single-allocator shard jobs must not require the full matrix.
            report = {
                "llm": f"{llm.config.provider}:{llm.config.model}",
                "lora_path": lora_path,
                "runner": runner,
                "only_allocator": only_allocator,
                "budget_fraction": budget_fraction
                if budget_fraction is not None
                else (batches[0].budget_fraction if batches else None),
                "live_min_per_task": live_min_per_task,
                only_allocator: row,
            }
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            typer.echo(f"Wrote single-allocator report -> {out}")
        return

    missing = [k for k in ALLOCATOR_KEYS if k not in rows]
    if missing and ckpt:
        for key in missing:
            shard = ckpt / f"{key}.json"
            if shard.is_file():
                rows[key] = json.loads(shard.read_text(encoding="utf-8"))["result"]
        missing = [k for k in ALLOCATOR_KEYS if k not in rows]
    if missing:
        raise typer.BadParameter(f"Missing allocator results: {', '.join(missing)}")

    report = _build_report(
        rows,
        llm=llm,
        lora_path=lora_path,
        runner=runner,
        scarcity_boost=scarcity_boost,
        shift_fraction=shift_fraction,
        swe_min_reserve=swe_min_reserve,
        roi_skip=roi_skip,
        fairness_reserve=fairness_reserve,
        hard_min_frac=hard_min_frac,
        batches=batches,
        bench_set=bench_set,
        budget_fraction=budget_fraction,
        live_min_per_task=live_min_per_task,
    )

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
