"""Tests for tool-aware GRPO rewards."""

from hbac.training.tool_reward import extract_tool_json, tool_aware_reward


def test_extract_valid_json():
    text = '{"thought": "ok", "tool_name": "bash", "tool_input": "ls"}'
    parsed = extract_tool_json(text)
    assert parsed is not None
    assert parsed["tool_name"] == "bash"


def test_tool_aware_reward_match():
    ref = '{"thought": "x", "tool_name": "submit", "tool_input": "done"}'
    comp = '{"thought": "y", "tool_name": "submit", "tool_input": "done"}'
    br = tool_aware_reward(comp, reference_completion=ref, reference_tool="submit")
    assert br.total >= 0.7


def test_tool_aware_reward_invalid_json():
    br = tool_aware_reward("not json at all", reference_completion='{"tool_name":"bash"}', reference_tool="bash")
    assert br.total < 0.3
