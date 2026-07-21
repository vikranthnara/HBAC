"""Lock canonical result artifacts and headline metrics for paper narrative."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(help="Generate locked canonical artifact manifest")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _hbac(data: dict) -> dict:
    r = data.get("hbac_joint", data.get("hbac_d18", data))
    return {
        "pass_at_1": r.get("pass_at_1"),
        "mean_batch_reward": r.get("mean_batch_reward"),
        "mean_tokens_used": r.get("mean_tokens_used"),
        "batch_violation_rate": r.get("batch_violation_rate"),
        "mean_parse_failures_per_task": r.get("mean_parse_failures_per_task"),
    }


@app.command()
def main(
    output: str = typer.Option("results/canonical_artifacts.json", help="Manifest output"),
) -> None:
    root = Path(".")
    live_legacy = root / "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"
    live_canon = root / "results/rivanna/compose_live_v3_heuristics_floor400_n2000_dpo_v2.json"
    live_path = live_canon if live_canon.is_file() else live_legacy
    live_data = _load(live_path) or {}

    manifest = {
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "V3 paper-ready canonical artifacts — conditional claims only",
        "thesis": (
            "Characterize when learned batch allocation matters under token scarcity; "
            "retract dominance claims where CIs overlap or capability gate fails"
        ),
        "claims": {
            "oracle_tier_a": {
                "artifact": "results/rivanna/v3_real_oracle_matrix.json",
                "claim": "HBAC D18/joint 80% vs Tier-A official 60%; ties type-prior at 80%",
                "metrics": _load(root / "results/rivanna/v3_real_oracle_matrix.json") or {},
            },
            "d18_oracle_ladder": {
                "artifact": "results/d18_oracle_ladder.json",
                "claim": "D18 without guardrail ties joint at 80% oracle; guardrail hurts to 74.7%",
                "verdict": (_load(root / "results/d18_oracle_ladder.json") or {}).get("verdict"),
            },
            "live_v3_d18_n2000": {
                "artifact": "results/rivanna/compose_live_v3_d18_floor400_n2000.json",
                "claim": (
                    "7B+DPO v2: hbac_d18 directionally higher than type-prior +1.3 pp "
                    "(27.65% vs 26.35%); paired McNemar p<1e-7; CI [0.85,1.85] pp; LCB-only"
                ),
                "paired_analysis": "results/paired_allocator_analysis_v3_d18.json",
                "analysis": "results/v3_d18_live_analysis.json",
            },
            "live_v3_directional_n2000": {
                "artifact": str(live_path.relative_to(root)),
                "claim": (
                    "7B baseline: hbac_guardrail directionally higher than type-prior "
                    "+1.3 pp (27.65% vs 26.35%); CIs overlap; LCB-only gap"
                ),
                "metrics": {
                    "hbac_guardrail": live_data.get("hbac_guardrail", live_data.get("hbac_fair", {})),
                    "hbac_d18": live_data.get("hbac_d18", {}),
                    "type_prior": live_data.get("type_prior", {}),
                },
                "paired_analysis": "results/paired_allocator_analysis.json",
            },
            "capability_gate": {
                "artifact": "results/capability_pilot_analysis.json",
                "claim": "7B and 32B fail SWE gate (0%); live scoped to LCB+tau+toolbench",
                "gate_passed": (_load(root / "results/capability_pilot_analysis.json") or {}).get(
                    "gate_passed"
                ),
            },
            "dpo_contamination": {
                "artifact": "results/dpo_contamination_audit.json",
                "claim": "v2 DPO: 100 exact LCB task-ID overlaps — FAIL; holdout retrain required",
                "verdict": (_load(root / "results/dpo_contamination_audit.json") or {}).get("verdict"),
            },
            "dpo_holdout_policy": {
                "artifact": "results/dpo_holdout_policy.json",
                "claim": "Benchmark-family exclusion yields 0 task-ID overlap by construction",
                "verdict": (_load(root / "results/dpo_holdout_policy.json") or {}).get("verdict"),
            },
            "dpo_holdout_retrain": {
                "artifact": "checkpoints/llm_dpo/20260710T045554Z_capability_holdout",
                "claim": (
                    "Holdout DPO LoRA trained (job 16854770); LCB exact overlap=0; "
                    "audit FAIL on residual SWE/tau/toolbench stub IDs vs eval_n1000"
                ),
                "audit": "results/dpo_contamination_audit_holdout.json",
                "verdict": (_load(root / "results/dpo_contamination_audit_holdout.json") or {}).get(
                    "verdict"
                ),
            },
            "live_v3_holdout_n2000": {
                "artifact": "results/rivanna/compose_live_v3_holdout_floor400_n2000.json",
                "claim": "Pending: V3 D18 live with LCB-holdout LoRA (primary allocators)",
                "paired_analysis": "results/paired_allocator_analysis_v3_holdout.json",
                "status": "gpu_pending",
            },
            "credit_beta_theory": {
                "artifact": "results/credit_beta_sweep.json",
                "claim": "Small beta (0.2) safe variance reduction; bias bound <= 0.018 on stub batches",
                "theory": "paper/appendix_theory.tex",
            },
            "paired_stats": {
                "artifact": "results/paired_allocator_analysis_v3_d18.json",
                "claim": "True per-task McNemar: +1.3 pp, p<1e-7, CI [0.85,1.85] pp",
                "verdict": (_load(root / "results/paired_allocator_analysis_v3_d18.json") or {}).get(
                    "verdict"
                ),
            },
            "power_analysis": {
                "artifact": "results/power_analysis_paired.json",
                "recommendation": (_load(root / "results/power_analysis_paired.json") or {}).get(
                    "recommendation"
                ),
            },
            "budget_share_ethics": {
                "artifact": "results/budget_share_starvation.json",
                "claim": "type_prior starve=1.0 / LCB share 0.1%; hbac_d18 LCB share 15.4%, starve=0",
            },
            "hard_min_frac_oracle_ablation": {
                "artifact": "results/hard_min_frac_oracle_sweep.json",
                "claim": "Guardrail never beats type-prior on oracle at hard_min_frac 0.10–0.25",
            },
        },
        "retracted_or_downgraded": [
            "hbac_fair beats type-prior (dominance) — use directionally higher + paired p-value",
            "Hierarchical POMDP framing — renamed decoupled batch allocator + stop classifier",
            "Compliant utility in main paper — appendix only",
            "Stub proxy baselines in main oracle table — Tier-A only in main text",
            "SWE heterogeneous-benchmark claim — scoped out until SWE pass@1 >= 5%",
        ],
        "gpu_pending": {
            "live_v3_holdout_matrix": "16972366 + 16972367 — holdout LoRA live",
            "capability_qwen3_4bit": "slurm/eval_capability_pilot_4bit.sh — SWE gate retry",
        },
        "gpu_completed": {
            "capability_pilot": "16846575 — 32B uniform n=60; SWE gate fails",
            "live_v3_d18_matrix": "16845485 + merge 16846574 — compose_live_v3_d18_floor400_n2000.json",
            "dpo_holdout_retrain": "16854770 — checkpoints/llm_dpo/20260710T045554Z_capability_holdout",
        },
        "reproduce": "scripts/reproduce_v3.sh",
        "five_of_five_ready": False,
        "five_of_five_blocker": "SWE pass@1 >= 5% on uniform capability pilot",
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
