"""Paired McNemar analysis for live allocator comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(help="Paired allocator analysis (McNemar, bootstrap CI)")


def _mcnemar_exact_p(b: int, c: int) -> float:
    if b + c == 0:
        return 1.0
    from math import comb

    n = b + c
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) * (0.5**n))


def _holm_bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        raw = pvals[idx]
        corrected = min(1.0, raw * (m - rank))
        running = max(running, corrected)
        adj[idx] = running
    return adj


def _load_vectors(path: Path, key: str | None) -> tuple[list[str], list[bool]] | None:
    data = json.loads(path.read_text())
    block = data.get(key, data) if key else data
    if "result" in block:
        block = block["result"]
    ids = block.get("per_task_ids")
    succ = block.get("per_task_successes")
    if ids and succ and len(ids) == len(succ):
        return ids, [bool(x) for x in succ]
    return None


def _compare(a: list[bool], b: list[bool], label_a: str, label_b: str) -> dict:
    b_cnt = sum(1 for x, y in zip(a, b) if x and not y)
    c_cnt = sum(1 for x, y in zip(a, b) if not x and y)
    n = len(a)
    diff_pp = (sum(a) - sum(b)) / max(n, 1) * 100
    rng = np.random.default_rng(0)
    boots = []
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        boots.append((arr_a[idx].mean() - arr_b[idx].mean()) * 100)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {
        "alloc_a": label_a,
        "alloc_b": label_b,
        "n": n,
        "pass_a": sum(a) / n,
        "pass_b": sum(b) / n,
        "gap_pp": diff_pp,
        "paired_ci95_pp": [lo, hi],
        "discordant_b": b_cnt,
        "discordant_c": c_cnt,
        "mcnemar_exact_p": _mcnemar_exact_p(b_cnt, c_cnt),
        "a_beats_b": diff_pp > 0,
    }


@app.command()
def main(
    shard_dir: str = typer.Option("results/rivanna/live_n2000_shards"),
    merged: str | None = typer.Option(
        "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"
    ),
    pairs: str = typer.Option(
        "hbac_fair:type_prior,hbac_joint:type_prior,hbac_d18:type_prior",
        help="Comma-separated alloc_a:alloc_b",
    ),
    output: str = typer.Option("results/paired_allocator_analysis.json"),
) -> None:
    shard_root = Path(shard_dir)
    comparisons: list[dict] = []
    pvals: list[float] = []
    notes: list[str] = []

    pair_specs = [p.strip() for p in pairs.split(",") if p.strip()]
    vectors: dict[str, tuple[list[str], list[bool]]] = {}

    for spec in pair_specs:
        a_name, b_name = spec.split(":", 1)
        for name in (a_name, b_name):
            if name in vectors:
                continue
            shard = shard_root / f"{name}.json"
            if shard.is_file():
                loaded = _load_vectors(shard, None)
                if loaded:
                    vectors[name] = loaded
                    continue
            if merged:
                merged_path = Path(merged)
                if merged_path.is_file():
                    loaded = _load_vectors(merged_path, name)
                    if loaded:
                        vectors[name] = loaded

    for spec in pair_specs:
        a_name, b_name = spec.split(":", 1)
        if a_name not in vectors or b_name not in vectors:
            notes.append(f"Missing per_task vectors for {a_name} vs {b_name}; run live eval with --save-per-task")
            if merged and Path(merged).is_file():
                data = json.loads(Path(merged).read_text())
                if a_name in data and b_name in data:
                    n = int(data[a_name].get("num_tasks", 2000))
                    pa = data[a_name]["pass_at_1"]
                    pb = data[b_name]["pass_at_1"]
                    sa = int(round(n * pa))
                    sb = int(round(n * pb))
                    b_est = max(0, sa - sb)
                    row = {
                        "alloc_a": a_name,
                        "alloc_b": b_name,
                        "n": n,
                        "pass_a": pa,
                        "pass_b": pb,
                        "gap_pp": (pa - pb) * 100,
                        "paired_ci95_pp": None,
                        "discordant_b": b_est,
                        "discordant_c": 0,
                        "mcnemar_exact_p": _mcnemar_exact_p(b_est, 0),
                        "method": "marginal_estimate",
                        "a_beats_b": pa > pb,
                    }
                    comparisons.append(row)
                    pvals.append(row["mcnemar_exact_p"])
            continue
        ids_a, succ_a = vectors[a_name]
        ids_b, succ_b = vectors[b_name]
        if ids_a != ids_b:
            notes.append(f"Task ID order mismatch {a_name} vs {b_name}")
            continue
        row = _compare(succ_a, succ_b, a_name, b_name)
        row["method"] = "paired_vectors"
        comparisons.append(row)
        pvals.append(row["mcnemar_exact_p"])

    if pvals:
        adj = _holm_bonferroni(pvals)
        for row, p_adj in zip(comparisons, adj):
            row["mcnemar_p_bonferroni"] = p_adj
            row["significant_005"] = p_adj < 0.05

    primary = next((c for c in comparisons if c["alloc_a"] in ("hbac_d18", "hbac_fair", "hbac_guardrail")), None)
    verdict = "INCONCLUSIVE"
    if primary:
        if primary.get("significant_005"):
            verdict = "DIRECTIONAL_SUPPORTED"
        elif primary.get("mcnemar_exact_p", 1) > 0.10:
            verdict = "RETRACT_BEATS_CLAIM"
        else:
            verdict = "TREND_ONLY"

    report = {
        "shard_dir": shard_dir,
        "merged": merged,
        "comparisons": comparisons,
        "notes": notes,
        "verdict": verdict,
        "preregistered": "research docs/Preregistered Analysis.md",
    }

    out = Path(output)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
