"""Oracle replay: compare baseline L1 vs D16 parse-penalty L1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from hbac.training.batch_curriculum import generate_curriculum_batches
from hbac.training.level1 import Level1Policy
from hbac.training.oracle_replay import OracleIndex
from hbac.training.phase3_pipeline import _resolve_l2, evaluate_l1_policy

app = typer.Typer(help="Oracle compare two L1 checkpoints")


@app.command()
def main(
    oracle_path: str = typer.Option("data/oracles", help="Oracle root"),
    l2_checkpoint: str = typer.Option("checkpoints/variant_a/latest", help="Frozen L2"),
    baseline_l1: str = typer.Option(..., help="Baseline L1 .npz"),
    candidate_l1: str = typer.Option(..., help="Candidate L1 .npz (e.g. D16 parse-penalty)"),
    num_batches: int = typer.Option(30, help="Eval batches"),
    seed: int = typer.Option(42, help="Batch seed"),
    output: str = typer.Option("results/d16_oracle_compare.json", help="Output JSON"),
) -> None:
    l2 = _resolve_l2(Path(l2_checkpoint))
    l1_base = Level1Policy.load(Path(baseline_l1))
    l1_cand = Level1Policy.load(Path(candidate_l1))
    oracle_index = OracleIndex(Path(oracle_path))

    batches = generate_curriculum_batches(Path(oracle_path), num_batches=num_batches, seed=seed)
    for b in batches:
        b.global_budget = max(len(b.tasks) * 40, int(b.oracle_token_sum * 0.40))
        b.budget_fraction = 0.40

    m_base = evaluate_l1_policy(l1_base, batches, l2, oracle_index, parse_penalty=0.0)
    m_cand = evaluate_l1_policy(l1_cand, batches, l2, oracle_index, parse_penalty=0.0)
    m_cand_pen = evaluate_l1_policy(l1_cand, batches, l2, oracle_index, parse_penalty=0.3)

    report = {
        "baseline_l1": baseline_l1,
        "candidate_l1": candidate_l1,
        "num_batches": len(batches),
        "baseline": m_base.to_dict(),
        "candidate_reward_fn": m_cand.to_dict(),
        "candidate_parse_penalty_0.3_reward_fn": m_cand_pen.to_dict(),
        "delta_pass_at_1_pp": (m_cand.pass_at_1 - m_base.pass_at_1) * 100,
        "delta_mean_reward": m_cand.mean_batch_reward - m_base.mean_batch_reward,
        "candidate_beats_baseline": m_cand.pass_at_1 > m_base.pass_at_1
        or m_cand.mean_batch_reward > m_base.mean_batch_reward,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
