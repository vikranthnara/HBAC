"""Merge per-allocator compose-live shards into one report JSON."""

from __future__ import annotations

import json
from pathlib import Path

import typer

REQUIRED_KEYS = (
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
    "hbac_fair",
)

V3_EXTRA_KEYS = ("hbac_d18", "hbac_guardrail")

app = typer.Typer(help="Merge compose-live allocator shards")


@app.command()
def main(
    shard_dir: str = typer.Option(..., help="Directory with per-allocator JSON shards"),
    output: str = typer.Option(..., help="Merged report output path"),
    meta_path: str | None = typer.Option(
        None, help="Optional meta.json written by shard jobs (run config)"
    ),
    v3: bool = typer.Option(False, help="Require hbac_d18 and hbac_guardrail shards"),
    keys: str = typer.Option(
        "",
        help="Comma-separated allocator keys to merge (empty = full REQUIRED set)",
    ),
    allow_partial: bool = typer.Option(
        False, help="Skip missing shards instead of failing (for primary-only holdout runs)"
    ),
) -> None:
    shard_root = Path(shard_dir)
    if not shard_root.is_dir():
        raise typer.BadParameter(f"Shard dir not found: {shard_root}")

    meta: dict = {}
    meta_file = Path(meta_path) if meta_path else shard_root / "meta.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))

    if keys.strip():
        key_list = [k.strip() for k in keys.split(",") if k.strip()]
    else:
        key_list = list(REQUIRED_KEYS) + (list(V3_EXTRA_KEYS) if v3 else [])
    rows: dict[str, dict] = {}
    missing: list[str] = []
    for key in key_list:
        shard = shard_root / f"{key}.json"
        if not shard.is_file():
            missing.append(key)
            continue
        payload = json.loads(shard.read_text(encoding="utf-8"))
        rows[key] = payload["result"] if "result" in payload else payload

    if missing and not allow_partial:
        raise typer.BadParameter(f"Missing shards: {', '.join(missing)}")

    hbac = rows.get("hbac_joint") or rows.get("hbac_d18") or next(iter(rows.values()))
    hbac_fair = rows.get("hbac_fair") or rows.get("hbac_guardrail", hbac)
    hbac_d18 = rows.get("hbac_d18", hbac)
    hbac_guardrail = rows.get("hbac_guardrail", hbac_fair)
    uniform = rows.get("uniform", {})
    clear_compose = rows.get("clear_compose", {})
    zebra_compose = rows.get("zebra_compose", {})
    type_prior = rows.get("type_prior", {})

    report = {
        "llm": meta.get("llm"),
        "lora_path": meta.get("lora_path"),
        "runner": meta.get("runner", "react"),
        "scarcity_boost": meta.get("scarcity_boost", False),
        "shift_fraction": meta.get("shift_fraction"),
        "swe_min_reserve": meta.get("swe_min_reserve"),
        "roi_skip": meta.get("roi_skip", False),
        "fairness_reserve": meta.get("fairness_reserve", False),
        "hard_min_frac": meta.get("hard_min_frac"),
        "num_tasks": meta.get("num_tasks") or hbac.get("num_tasks"),
        "benchmarks": meta.get("benchmarks", []),
        "budget_fraction": meta.get("budget_fraction"),
        "live_min_per_task": meta.get("live_min_per_task"),
        **rows,
        "hbac_beats_clear": bool(clear_compose)
        and (
            hbac["pass_at_1"] > clear_compose["pass_at_1"]
            or hbac["mean_batch_reward"] > clear_compose["mean_batch_reward"]
        ),
        "hbac_beats_uniform": bool(uniform)
        and (
            hbac["pass_at_1"] > uniform["pass_at_1"]
            or hbac["mean_batch_reward"] > uniform["mean_batch_reward"]
        ),
        "hbac_beats_zebra": bool(zebra_compose)
        and (
            hbac["pass_at_1"] > zebra_compose["pass_at_1"]
            or hbac["mean_batch_reward"] > zebra_compose["mean_batch_reward"]
        ),
        "hbac_fair_beats_type_prior": bool(type_prior)
        and hbac_fair["pass_at_1"] > type_prior["pass_at_1"],
        "hbac_d18_beats_type_prior": bool(type_prior)
        and hbac_d18["pass_at_1"] > type_prior["pass_at_1"],
        "hbac_guardrail_beats_type_prior": bool(type_prior)
        and hbac_guardrail["pass_at_1"] > type_prior["pass_at_1"],
        "proxy_disclaimer": (
            "clear_official/zebra_official/reforc_official: Tier-A paper algorithms. "
            "Heuristics are reviewer baselines."
        ),
        "merged_from_shards": sorted(rows.keys()),
        "missing_shards": missing,
    }

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    typer.echo(f"Merged {len(rows)} shards -> {out}")
    typer.echo(f"hbac_fair pass@1={hbac_fair['pass_at_1']:.4f} type_prior={type_prior['pass_at_1']:.4f}")


if __name__ == "__main__":
    app()
