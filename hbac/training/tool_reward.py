"""Tool-JSON rewards aligned with live eval (BaseRunner.parse_action)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hbac.baselines.react import ReActRunner


def extract_tool_json(text: str) -> dict | None:
    text = text.strip()
    candidates: list[dict] = []
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text):
        try:
            data = json.loads(match.group())
            if isinstance(data, dict):
                candidates.append(data)
        except json.JSONDecodeError:
            continue
    for data in reversed(candidates):
        if data.get("tool_name") or data.get("action"):
            return data
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


@dataclass
class ToolRewardBreakdown:
    total: float
    valid_json: float
    tool_name_match: float
    tool_input_match: float
    overlap: float


def tool_aware_reward(
    completion: str,
    *,
    reference_completion: str = "",
    reference_tool: str | None = None,
    success_weight: float = 1.0,
) -> ToolRewardBreakdown:
    """Reward completions that match live-eval tool JSON format."""
    parsed = extract_tool_json(completion)
    valid = 0.4 if parsed else 0.0
    name_match = 0.0
    input_match = 0.0

    if parsed:
        comp_tool = str(parsed.get("tool_name") or parsed.get("action") or "")
        if reference_tool and comp_tool == reference_tool:
            name_match = 0.3
        elif reference_tool is None and comp_tool:
            name_match = 0.15

        ref_parsed = extract_tool_json(reference_completion) if reference_completion else None
        if ref_parsed:
            ref_in = str(ref_parsed.get("tool_input", ref_parsed.get("command", "")) or "")
            comp_in = str(parsed.get("tool_input", parsed.get("command", "")) or "")
            if ref_in and comp_in and ref_in.strip() == comp_in.strip():
                input_match = 0.2
            elif ref_in and comp_in:
                overlap_in = sum(1 for a, b in zip(comp_in, ref_in) if a == b) / max(len(ref_in), 1)
                input_match = 0.1 * overlap_in

    ref = reference_completion.strip()
    comp = completion.strip()
    overlap = 0.0
    if ref and comp:
        overlap = 0.1 * min(1.0, sum(1 for a, b in zip(comp, ref) if a == b) / max(len(ref), 1))

    total = (valid + name_match + input_match + overlap) * success_weight
    return ToolRewardBreakdown(
        total=min(total, 1.0),
        valid_json=valid,
        tool_name_match=name_match,
        tool_input_match=input_match,
        overlap=overlap,
    )


def build_chat_prompt(
    benchmark: str,
    *,
    user_content: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    system = ReActRunner.system_prompt_for_benchmark(benchmark)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    if user_content:
        messages.append({"role": "user", "content": user_content})
    return messages


def format_prompt_for_trl(messages: list[dict[str, str]], tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)
