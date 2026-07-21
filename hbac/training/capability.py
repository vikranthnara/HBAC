"""TRACE-inspired capability analysis and DPO pair construction (Phase 3c)."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from hbac.core.trajectory import TrajectoryStore
from hbac.training.dataset import find_oracle_paths
from hbac.training.grpo_records import load_grpo_step_records
from hbac.training.tool_reward import extract_tool_json, tool_aware_reward


@dataclass
class CapabilityDeficit:
    capability_id: str
    description: str
    benchmark: str
    fail_count: int
    success_count: int
    example_task_ids: list[str] = field(default_factory=list)


@dataclass
class DpoPair:
    prompt: str
    chosen: str
    rejected: str
    capability_id: str
    benchmark: str
    task_id: str
    chosen_reward: float
    rejected_reward: float


def _failure_tags(step_text: str, *, reference_tool: str | None) -> list[str]:
    tags: list[str] = []
    parsed = extract_tool_json(step_text)
    if parsed is None:
        tags.append("valid_tool_json")
        return tags
    tool = str(parsed.get("tool_name") or parsed.get("action") or "")
    if reference_tool and tool != reference_tool:
        tags.append("tool_name_match")
    if not tool:
        tags.append("valid_tool_json")
    return tags


def analyze_capability_deficits(oracle_root: Path) -> list[CapabilityDeficit]:
    """Contrast successful vs failed oracle trajectories per benchmark."""
    stats: dict[tuple[str, str], dict] = {}

    for path in find_oracle_paths(oracle_root):
        store = TrajectoryStore(path)
        for traj in store.load_all():
            label = "success" if traj.success else "fail"
            for step in traj.steps:
                if not step.llm_response:
                    continue
                ref_tool = step.action.tool_name if step.action else None
                for cap in _failure_tags(step.llm_response, reference_tool=ref_tool):
                    key = (traj.benchmark, cap)
                    bucket = stats.setdefault(
                        key,
                        {"fail": 0, "success": 0, "tasks": set()},
                    )
                    bucket[label] += 1
                    bucket["tasks"].add(traj.task_id)

    out: list[CapabilityDeficit] = []
    for (benchmark, cap_id), bucket in sorted(stats.items()):
        if bucket["fail"] == 0:
            continue
        out.append(
            CapabilityDeficit(
                capability_id=cap_id,
                description=f"{cap_id} on {benchmark}",
                benchmark=benchmark,
                fail_count=int(bucket["fail"]),
                success_count=int(bucket["success"]),
                example_task_ids=sorted(bucket["tasks"])[:5],
            )
        )
    return out


def _corrupt_completion(completion: str, *, mode: str) -> str:
    if mode == "invalid_json":
        return completion.replace("{", "").replace("}", "")[:120] or "not json"
    if mode == "wrong_tool":
        return re.sub(
            r'"tool_name"\s*:\s*"[^"]+"',
            '"tool_name": "wrong_tool"',
            completion,
            count=1,
        )
    if mode == "empty_tool":
        return '{"tool_name": "", "tool_input": ""}'
    return completion + " trailing garbage"


def build_dpo_pairs(
    oracle_root: Path,
    *,
    limit: int = 400,
    tokenizer=None,
    reject_modes: tuple[str, ...] = ("wrong_tool", "invalid_json"),
    benchmark: str | None = None,
    oversample_benchmark: str | None = None,
    oversample_factor: int = 1,
    exclude_task_ids: set[str] | None = None,
    exclude_benchmarks: tuple[str, ...] | None = None,
) -> list[DpoPair]:
    """Build preference pairs: oracle completion (chosen) vs synthetic failures (rejected)."""
    from hbac.training.grpo_records import format_prompt_for_trl

    excluded = exclude_task_ids or set()
    skip_benches = set(exclude_benchmarks or ())
    records = load_grpo_step_records(
        oracle_root,
        limit=limit * (20 if (excluded or skip_benches) else 2),
        successful_only=True,
        benchmark=benchmark,
    )
    pairs: list[DpoPair] = []

    for row in records:
        if not row.get("reference_tool"):
            continue
        task_id = str(row.get("task_id", ""))
        bench = str(row.get("benchmark", ""))
        if task_id in excluded or bench in skip_benches:
            continue
        prompt = (
            format_prompt_for_trl(row["messages"], tokenizer)
            if tokenizer
            else json.dumps(row["messages"])
        )
        chosen = row["completion"]
        ref_tool = row.get("reference_tool")
        chosen_r = tool_aware_reward(
            chosen,
            reference_completion=chosen,
            reference_tool=ref_tool,
            success_weight=float(row.get("reward_weight", 1.0)),
        ).total

        for mode in reject_modes:
            rejected = _corrupt_completion(chosen, mode=mode)
            rejected_r = tool_aware_reward(
                rejected,
                reference_completion=chosen,
                reference_tool=ref_tool,
                success_weight=float(row.get("reward_weight", 1.0)),
            ).total
            if rejected_r >= chosen_r:
                continue
            pairs.append(
                DpoPair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    capability_id=mode,
                    benchmark=str(row.get("benchmark", "")),
                    task_id=str(row.get("task_id", "")),
                    chosen_reward=chosen_r,
                    rejected_reward=rejected_r,
                )
            )

    pairs.sort(key=lambda p: p.chosen_reward - p.rejected_reward, reverse=True)
    if oversample_benchmark and oversample_factor > 1:
        extra = [p for p in pairs if p.benchmark == oversample_benchmark]
        pairs = extra * (oversample_factor - 1) + pairs
        pairs.sort(key=lambda p: p.chosen_reward - p.rejected_reward, reverse=True)
    return pairs[:limit]


def write_capability_report(oracle_root: Path, output: Path, *, pair_limit: int = 200) -> dict:
    deficits = analyze_capability_deficits(oracle_root)
    pairs = build_dpo_pairs(oracle_root, limit=pair_limit, reject_modes=("wrong_tool",))
    by_cap: dict[str, int] = {}
    for p in pairs:
        by_cap[p.capability_id] = by_cap.get(p.capability_id, 0) + 1
    report = {
        "oracle_root": str(oracle_root),
        "deficits": [asdict(d) for d in deficits],
        "primary_capability": deficits[0].capability_id if deficits else "valid_tool_json",
        "dpo_pairs": {
            "total": len(pairs),
            "by_capability": by_cap,
            "mean_margin": float(
                sum(p.chosen_reward - p.rejected_reward for p in pairs) / max(len(pairs), 1)
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
