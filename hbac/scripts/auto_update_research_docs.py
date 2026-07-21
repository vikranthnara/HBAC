"""Auto-update research docs from latest Rivanna pull artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_MD = ROOT / "research docs" / "Results.md"
DISCOVERY_MD = ROOT / "research docs" / "Research Discovery.md"
SUMMARY = ROOT / "results" / "experiment_summary.json"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def update_floor400_baselines() -> bool:
    path = ROOT / "results/rivanna/compose_live_bf040_floor400_all_baselines_dpo_v2.json"
    data = _load(path)
    if not data:
        return False
    hb = data.get("hbac_joint", {})
    uni = data.get("uniform", {})
    cl = data.get("clear_compose", {})
    zb = data.get("zebra_compose", {})
    block = (
        f"\n#### Live all-baselines @ floor=400 (job `16771884_1`)\n\n"
        f"| Allocator | pass@1 | Mean reward | Tokens | Violations |\n"
        f"|-----------|--------|-------------|--------|------------|\n"
        f"| **HBAC** | {_fmt_pct(hb.get('pass_at_1', 0))} | **{hb.get('mean_batch_reward', 0):+.2f}** | "
        f"**{hb.get('mean_tokens_used', 0):.0f}** | **{_fmt_pct(hb.get('batch_violation_rate', 0))}** |\n"
        f"| Uniform | {_fmt_pct(uni.get('pass_at_1', 0))} | {uni.get('mean_batch_reward', 0):+.2f} | "
        f"{uni.get('mean_tokens_used', 0):.0f} | {_fmt_pct(uni.get('batch_violation_rate', 0))} |\n"
        f"| CLEAR | {_fmt_pct(cl.get('pass_at_1', 0))} | {cl.get('mean_batch_reward', 0):+.2f} | "
        f"{cl.get('mean_tokens_used', 0):.0f} | {_fmt_pct(cl.get('batch_violation_rate', 0))} |\n"
        f"| ZEBRA | {_fmt_pct(zb.get('pass_at_1', 0))} | {zb.get('mean_batch_reward', 0):+.2f} | "
        f"{zb.get('mean_tokens_used', 0):.0f} | {_fmt_pct(zb.get('batch_violation_rate', 0))} |\n"
    )
    text = RESULTS_MD.read_text(encoding="utf-8")
    marker = "#### Live all-baselines @ floor=400"
    if marker in text:
        start = text.index(marker)
        end = text.find("\n#### ", start + 1)
        if end == -1:
            end = text.find("\n### ", start + 1)
        if end == -1:
            end = len(text)
        text = text[:start] + block.strip() + "\n" + text[end:]
    else:
        anchor = "Floor=400 all-baselines eval (`16771884_1`) pending at pull time."
        if anchor in text:
            text = text.replace(anchor, block.strip())
        else:
            anchor2 = "#### Live dual-regime"
            text = text.replace(anchor2, block.strip() + "\n\n" + anchor2, 1)
    RESULTS_MD.write_text(text, encoding="utf-8")
    return True


def update_d14_roi() -> bool:
    path = ROOT / "results/rivanna/compose_live_bf040_floor300_roi_skip_dpo_v2.json"
    data = _load(path)
    if not data:
        return False
    hb = data["hbac_joint"]
    uni = data["uniform"]
    gap = (hb["pass_at_1"] - uni["pass_at_1"]) * 100
    note = (
        f"**D14 ROI skip (floor=300):** HBAC {_fmt_pct(hb['pass_at_1'])} vs uniform "
        f"{_fmt_pct(uni['pass_at_1'])} (**+{gap:.1f} pp**); reward {hb['mean_batch_reward']:.2f} vs "
        f"{uni['mean_batch_reward']:.2f}."
    )
    text = RESULTS_MD.read_text(encoding="utf-8")
    old = "**D12 scarcity boost (floor=400):**"
    if note not in text and old in text:
        text = text.replace(old, note + "\n\n" + old, 1)
        RESULTS_MD.write_text(text, encoding="utf-8")
    return True


def update_d12_refined() -> bool:
    refined_path = ROOT / "results/rivanna/compose_live_bf040_floor400_scarcity_refined_dpo_v2.json"
    baseline_path = ROOT / "results/rivanna/compose_live_bf040_floor400_dpo_v2.json"
    original_d12_path = ROOT / "results/rivanna/compose_live_bf040_floor400_scarcity_boost_dpo_v2.json"
    refined = _load(refined_path)
    if not refined:
        return False
    hb = refined.get("hbac_joint", {})
    base = (_load(baseline_path) or {}).get("hbac_joint", {})
    orig = (_load(original_d12_path) or {}).get("hbac_joint", {})
    base_rew = float(base.get("mean_batch_reward") or 0)
    orig_rew = float(orig.get("mean_batch_reward") or 0)
    rew = float(hb.get("mean_batch_reward") or 0)
    parse_f = float(hb.get("mean_parse_failures_per_task") or 0)
    base_parse = float(base.get("mean_parse_failures_per_task") or 0)
    orig_parse = float(orig.get("mean_parse_failures_per_task") or 0)
    swe_json = (
        hb.get("per_benchmark", {}).get("swe_bench", {}).get("first_step_valid_json_rate", 0) or 0
    )
    verdict = "improved" if parse_f < orig_parse and parse_f <= base_parse + 0.05 else "partial"
    if parse_f <= base_parse + 0.01 and rew > base_rew:
        verdict = "confirmed"
    note = (
        f"**D12 refined (job `16787709`):** shift=0.08, reserve=0.5 — pass@1 {_fmt_pct(hb.get('pass_at_1', 0))}, "
        f"reward {rew:.2f} (baseline {base_rew:.2f}, original D12 {orig_rew:.2f}), "
        f"parse {parse_f:.2f}/task (baseline {base_parse:.2f}, original D12 {orig_parse:.2f}), "
        f"SWE valid JSON {_fmt_pct(swe_json)}. Verdict: **{verdict}**."
    )
    text = RESULTS_MD.read_text(encoding="utf-8")
    pending = "**D12 refined (job `16787709`):** `shift_fraction=0.08`, `swe_min_reserve=0.5` — submitted to test parse guard; pending."
    old_block = "**D12 refined (job `16787709`):**"
    if pending in text:
        text = text.replace(pending, note)
    elif old_block in text:
        start = text.index(old_block)
        end = text.find("\n\n", start)
        if end == -1:
            end = len(text)
        text = text[:start] + note + text[end:]
    else:
        anchor = "**D12 scarcity boost (floor=400):**"
        text = text.replace(anchor, note + "\n\n" + anchor, 1)
    RESULTS_MD.write_text(text, encoding="utf-8")

    if DISCOVERY_MD.is_file():
        dt = DISCOVERY_MD.read_text(encoding="utf-8")
        dt = dt.replace(
            "| **16787709** | D12 refined scarcity boost @ floor=400 | RUNNING |",
            "| **16787709** | D12 refined scarcity boost @ floor=400 | ✅ COMPLETED |",
        )
        dt = dt.replace(
            "| **P1** | Refine D12 (`shift_fraction=0.08`, `swe_min_reserve=0.5`) | 🔄 Job `16787709` RUNNING |",
            "| **P1** | Refine D12 (`shift_fraction=0.08`, `swe_min_reserve=0.5`) | ✅ COMPLETED |",
        )
        DISCOVERY_MD.write_text(dt, encoding="utf-8")

    if SUMMARY.is_file():
        d = json.loads(SUMMARY.read_text())
        d["d12_refined"] = {
            "job": "16787709",
            "file": str(refined_path.relative_to(ROOT)),
            "pass_at_1": hb.get("pass_at_1"),
            "mean_batch_reward": rew,
            "mean_parse_failures_per_task": parse_f,
            "swe_valid_json_rate": swe_json,
            "verdict": verdict,
            "status": "completed",
        }
        SUMMARY.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return True


def update_discovery_status() -> None:
  if not DISCOVERY_MD.is_file():
    return
  text = DISCOVERY_MD.read_text(encoding="utf-8")
  text = text.replace(
    "| **16771884_1** | P0 all baselines floor=400 | 🔄 RUNNING |",
    "| **16771884_1** | P0 all baselines floor=400 | ✅ COMPLETED |",
  )
  for job, label in [("16776563", "D16"), ("16776564", "D14")]:
    text = text.replace(
      f"| **{job}** | {label} | 🔄 Submitted |",
      f"| **{job}** | {label} | ✅ COMPLETED |",
    )
  DISCOVERY_MD.write_text(text, encoding="utf-8")


def update_v3_live() -> bool:
    path = ROOT / "results" / "v3_live_analysis.json"
    if not path.is_file():
        live = ROOT / "results" / "rivanna" / "compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"
        pilot = ROOT / "results" / "rivanna" / "compose_live_v3_pilot_floor400_dpo_v2.json"
        src = live if live.is_file() else pilot
        if not src.is_file():
            return False
        import subprocess
        subprocess.run(
            [
                "python",
                "-m",
                "hbac.scripts.analyze_v3_live",
                "--result-path",
                str(src),
                "--output",
                str(path),
            ],
            check=False,
            cwd=ROOT,
        )
    data = _load(path)
    if not data:
        return False
    block = (
        f"\n#### V3 live heuristic @ floor=400\n\n"
        f"- **hbac_fair_beats_type_prior:** {data.get('hbac_fair_beats_type_prior')}\n"
        f"- HBAC fair pass@1: {_fmt_pct(data.get('hbac_fair_pass_at_1', 0))} "
        f"(CI {data.get('allocators', {}).get('hbac_fair', {}).get('pass_at_1_ci95', '—')})\n"
        f"- Type-prior pass@1: {_fmt_pct(data.get('type_prior_pass_at_1', 0))}\n"
        f"- Gap: {data.get('hbac_fair_minus_type_prior_pp', 0):+.1f} pp\n"
        f"- Verdict: **{data.get('verdict')}**\n"
        f"- Source: `{data.get('source', '')}`\n"
    )
    text = RESULTS_MD.read_text(encoding="utf-8")
    marker = "#### V3 live heuristic @ floor=400"
    if marker in text:
        start = text.index(marker)
        end = text.find("\n#### ", start + 1)
        if end == -1:
            end = text.find("\n---", start + 1)
        if end == -1:
            end = len(text)
        text = text[:start] + block.strip() + "\n" + text[end:]
    else:
        anchor = "## 0b. V3 wave"
        if anchor in text:
            text = text.replace(anchor, block.strip() + "\n\n" + anchor, 1)
    RESULTS_MD.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    changed = []
    if update_floor400_baselines():
        changed.append("floor400_baselines")
    if update_d14_roi():
        changed.append("d14_roi")
    if update_d12_refined():
        changed.append("d12_refined")
    if update_v3_live():
        changed.append("v3_live")
    if changed:
        update_discovery_status()
    if SUMMARY.is_file():
        d = json.loads(SUMMARY.read_text())
        d["updated_at"] = datetime.now(timezone.utc).isoformat()
        d["auto_doc_refresh"] = changed
        SUMMARY.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"updated": changed}))


if __name__ == "__main__":
    main()
