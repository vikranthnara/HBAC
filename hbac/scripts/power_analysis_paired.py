"""Power analysis for paired allocator comparisons (McNemar)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(help="Paired McNemar power analysis for allocator A vs B")


def _discordant_from_vectors(a: list[bool], b: list[bool]) -> tuple[int, int, int]:
    if len(a) != len(b):
        raise ValueError(f"Length mismatch: {len(a)} vs {len(b)}")
    b_cnt = sum(1 for x, y in zip(a, b) if x and not y)
    c_cnt = sum(1 for x, y in zip(a, b) if not x and y)
    return b_cnt, c_cnt, len(a)


def _discordant_from_marginals(n: int, p_a: float, p_b: float, *, assume_subset: bool = True) -> tuple[int, int]:
    """Estimate (b, c) from marginals when per-task vectors unavailable."""
    s_a = int(round(n * p_a))
    s_b = int(round(n * p_b))
    if assume_subset and s_a >= s_b:
        # Optimistic: all B successes ⊆ A successes (observed for LCB-only gap)
        b = s_a - s_b
        c = 0
    else:
        overlap = min(s_a, s_b)
        b = s_a - overlap
        c = s_b - overlap
    return max(b, 0), max(c, 0)


def _mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
  # two-sided exact binomial
    from math import comb

    n = b + c
    k = min(b, c)
    p_one = sum(comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * p_one)


def _power_mcnemar(b_rate: float, n: int, alpha: float = 0.05) -> float:
    """Approximate power via normal approximation to McNemar (b vs c)."""
    if b_rate <= 0 or b_rate >= 0.5:
        return 0.05 if b_rate <= 0 else 0.99
    # E[b] = n * p_disc, with p_disc ≈ 2 * b_rate for symmetric c≈0
    exp_b = n * b_rate
    exp_c = n * b_rate * 0.05  # assume c << b
    var = exp_b + exp_c
    if var <= 0:
        return 0.05
    z_alpha = 1.96
    effect = exp_b - exp_c
    z = effect / math.sqrt(var)
    # P(Z > z_alpha - z)
    from statistics import NormalDist

    return 1.0 - NormalDist().cdf(z_alpha - z)


@app.command()
def main(
    source: str = typer.Option(
        "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json",
        help="Live eval JSON with allocator blocks",
    ),
    alloc_a: str = typer.Option("hbac_fair", help="Allocator A (or hbac_d18)"),
    alloc_b: str = typer.Option("type_prior", help="Allocator B"),
    per_task_a: str | None = typer.Option(None, help="JSON with per_task_successes for A"),
    per_task_b: str | None = typer.Option(None, help="JSON with per_task_successes for B"),
    target_power: float = typer.Option(0.80, help="Target power"),
    output: str = typer.Option("results/power_analysis_paired.json"),
) -> None:
    root = Path(source)
    data = json.loads(root.read_text()) if root.is_file() else {}

    b_cnt = c_cnt = n = 0
    method = "marginal_estimate"

    if per_task_a and per_task_b:
        va = json.loads(Path(per_task_a).read_text())
        vb = json.loads(Path(per_task_b).read_text())
        sa = va.get("per_task_successes") or va.get("result", {}).get("per_task_successes")
        sb = vb.get("per_task_successes") or vb.get("result", {}).get("per_task_successes")
        if sa and sb:
            b_cnt, c_cnt, n = _discordant_from_vectors(sa, sb)
            method = "paired_vectors"
    elif alloc_a in data and alloc_b in data:
        pa = data[alloc_a].get("pass_at_1", 0)
        pb = data[alloc_b].get("pass_at_1", 0)
        n = int(data[alloc_a].get("num_tasks") or data.get("num_tasks") or 2000)
        b_cnt, c_cnt = _discordant_from_marginals(n, pa, pb, assume_subset=True)
        method = "marginal_subset_estimate"

    disc_rate = (b_cnt + c_cnt) / max(n, 1)
    b_rate = b_cnt / max(n, 1)
    p_obs = _mcnemar_exact_p(b_cnt, c_cnt)

    # Search n for target power
    rec_n = n
    for trial_n in range(n, n * 8 + 1, max(1, n // 10)):
        if _power_mcnemar(b_rate, trial_n) >= target_power:
            rec_n = trial_n
            break
    else:
        rec_n = n * 8

    recommendation = "stop_publish_negative"
    if p_obs < 0.05:
        recommendation = "claim_directional_supported"
    elif p_obs < 0.10 and rec_n <= 5000:
        recommendation = "scale_to_recommended_n"
    elif rec_n > 5000:
        recommendation = "stop_publish_negative"

    report = {
        "source": str(source),
        "alloc_a": alloc_a,
        "alloc_b": alloc_b,
        "method": method,
        "n_observed": n,
        "discordant_b": b_cnt,
        "discordant_c": c_cnt,
        "mcnemar_exact_p": p_obs,
        "pass_a": data.get(alloc_a, {}).get("pass_at_1") if data else None,
        "pass_b": data.get(alloc_b, {}).get("pass_at_1") if data else None,
        "gap_pp": (
            (data.get(alloc_a, {}).get("pass_at_1", 0) - data.get(alloc_b, {}).get("pass_at_1", 0)) * 100
            if data
            else None
        ),
        "recommended_n_for_power": rec_n,
        "target_power": target_power,
        "recommendation": recommendation,
        "futility_rule": "If McNemar p > 0.10 at n=5000, retract beats-type-prior claim",
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
