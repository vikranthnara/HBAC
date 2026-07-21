"""Per-benchmark budget share + starvation-rate analysis for ethics / fairness tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer

from hbac.baselines.clear import CLEARAllocator
from hbac.baselines.clear_official import CLEAROfficialAllocator
from hbac.baselines.heuristics import SJFAllocator, TypePriorAllocator
from hbac.baselines.reforc_official import ReFORCOfficialAllocator
from hbac.baselines.zebra import ZEBRAAllocator
from hbac.baselines.zebra_official import ZEBRAOfficialAllocator
from hbac.scripts.eval_compose_live import _filter_batches
from hbac.training.batch_curriculum import load_batches
from hbac.training.l1_batch_reward import starvation_rate
from hbac.training.level1 import Level1Allocator, Level1Policy
from hbac.training.scarcity import fairness_reserve_alloc

app = typer.Typer(help="Budget share and starvation metrics across allocators")


@app.command()
def main(
    batches_path: str = typer.Option("checkpoints/eval_n1000/batches.jsonl"),
    l1_checkpoint: str = typer.Option(
        "checkpoints/variant_b/local_tight_bf040/stage3/20260701T175308Z/level1_policy.npz"
    ),
    l1_checkpoint_d18: str = typer.Option(
        "checkpoints/phase3_fairness_0.5/20260706T220026Z/stage3/level1_policy.npz"
    ),
    live_min_per_task: int = typer.Option(400),
    budget_fraction: float = typer.Option(0.4),
    hard_min_frac: float = typer.Option(0.15),
    output: str = typer.Option("results/budget_share_starvation.json"),
) -> None:
    raw = load_batches(Path(batches_path))
    batches = _filter_batches(
        raw,
        live=True,
        benchmarks={"livecodebench", "swe_bench", "tau_bench", "toolbench"},
        max_batches=0,
        live_min_per_task=live_min_per_task,
        budget_fraction_override=budget_fraction,
    )
    l1 = Level1Policy.load(Path(l1_checkpoint))
    l1_d18 = Level1Policy.load(Path(l1_checkpoint_d18))
    clear = CLEARAllocator(min_per_task=live_min_per_task)
    zebra = ZEBRAAllocator(min_per_task=live_min_per_task)
    clear_official = CLEAROfficialAllocator(min_per_task=live_min_per_task)
    zebra_official = ZEBRAOfficialAllocator(min_per_task=live_min_per_task)
    reforc_official = ReFORCOfficialAllocator(min_per_task=live_min_per_task)

    def hbac(b):
        return l1.allocate_schema(b, int(np.argmax(l1.schema_probs(b))))

    def d18(b):
        return l1_d18.allocate_schema(b, int(np.argmax(l1_d18.schema_probs(b))))

    def guard(b):
        return fairness_reserve_alloc(hbac(b), b, hard_min_frac=hard_min_frac)

    registry = {
        "uniform": lambda b: Level1Allocator(b.global_budget).allocate(b.task_ids),
        "type_prior": lambda b: TypePriorAllocator().allocate(b.tasks, b.global_budget),
        "hbac_joint": hbac,
        "hbac_d18": d18,
        "hbac_guardrail": guard,
        "sjf_compose": lambda b: SJFAllocator(min_per_task=1).allocate(b.tasks, b.global_budget),
        "clear_compose": lambda b: clear.allocate(b.tasks, b.global_budget),
        "zebra_compose": lambda b: zebra.allocate(b.tasks, b.global_budget),
        "clear_official": lambda b: clear_official.allocate(b.tasks, b.global_budget),
        "zebra_official": lambda b: zebra_official.allocate(b.tasks, b.global_budget),
        "reforc_official": lambda b: reforc_official.allocate(b.tasks, b.global_budget),
    }

    primary = ("hbac_d18", "type_prior", "hbac_guardrail", "uniform")
    out_rows: dict = {}
    for name in primary:
        fn = registry[name]
        bench_tok: dict[str, list[float]] = defaultdict(list)
        bench_share: dict[str, list[float]] = defaultdict(list)
        starve: list[float] = []
        zero_hard: list[float] = []
        for b in batches:
            alloc = fn(b)
            tot = max(sum(alloc.values()), 1)
            starve.append(starvation_rate(alloc, b, hard_min_frac=hard_min_frac))
            hard_ids = [t.task_id for t in b.tasks if t.benchmark in {"livecodebench", "swe_bench"}]
            zero_hard.append(
                sum(1 for tid in hard_ids if alloc.get(tid, 0) <= 1) / max(len(hard_ids), 1)
            )
            by_bench: dict[str, float] = defaultdict(float)
            for t in b.tasks:
                by_bench[t.benchmark] += alloc.get(t.task_id, 0)
            for bench, tok in by_bench.items():
                bench_tok[bench].append(tok)
                bench_share[bench].append(tok / tot)

        out_rows[name] = {
            "mean_starvation_rate": float(np.mean(starve)) if starve else 0.0,
            "mean_hard_zero_frac": float(np.mean(zero_hard)) if zero_hard else 0.0,
            "mean_budget_share_by_benchmark": {
                k: float(np.mean(v)) for k, v in sorted(bench_share.items())
            },
            "mean_tokens_by_benchmark": {
                k: float(np.mean(v)) for k, v in sorted(bench_tok.items())
            },
        }

    report = {
        "batches_path": batches_path,
        "num_batches": len(batches),
        "live_min_per_task": live_min_per_task,
        "budget_fraction": budget_fraction,
        "allocators": out_rows,
        "ethics_note": (
            "type_prior zeros hard-task budget by design; "
            "hbac_d18/guardrail retain residual LCB/SWE share under scarcity"
        ),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
