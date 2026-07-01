"""Hard-coded deterministic episodes for all four environment wrappers."""

from __future__ import annotations

from dataclasses import dataclass

from hbac.baselines.react import ReActRunner
from hbac.envs.livecodebench import LiveCodeBenchEnv
from hbac.envs.swe_bench import SWEBenchEnv
from hbac.envs.toolbench import ToolBenchEnv
from hbac.envs.tau_bench import TauBenchEnv


@dataclass
class DeterministicEpisode:
    env_name: str
    task_id: str
    system_prompt: str
    responses: list[str]


DETERMINISTIC_EPISODES: list[DeterministicEpisode] = [
    DeterministicEpisode(
        "swe_bench",
        "swe-local-1",
        ReActRunner.system_prompt_for_benchmark("swe_bench"),
        [
            '{"tool_name": "bash", "tool_input": "echo fix > foo.py"}',
            '{"tool_name": "submit", "tool_input": "diff --git a/foo.py b/foo.py\\n+fix\\n"}',
        ],
    ),
    DeterministicEpisode(
        "livecodebench",
        "lcb-local-1",
        ReActRunner.system_prompt_for_benchmark("livecodebench"),
        [
            '{"tool_name": "generate_code", "tool_input": "a=int(input())\\nb=int(input())\\nprint(a+b)"}',
            '{"tool_name": "run_tests", "tool_input": ""}',
        ],
    ),
    DeterministicEpisode(
        "toolbench",
        "toolbench-local-1",
        '{"tool_name": "list_apis"} uses list_apis, call_api, submit for API tasks.',
        [
            '{"tool_name": "list_apis", "tool_input": ""}',
            '{"tool_name": "call_api", "tool_input": "weather_api NYC"}',
            '{"tool_name": "submit", "tool_input": "72F"}',
        ],
    ),
    DeterministicEpisode(
        "tau_bench",
        "tau-local-1",
        "Use lookup, message_user, and submit for user interactions.",
        [
            '{"tool_name": "lookup", "tool_input": "flight AA100"}',
            '{"tool_name": "message_user", "tool_input": "Book AA100 tomorrow?"}',
            '{"tool_name": "submit", "tool_input": "confirmed"}',
        ],
    ),
]


def make_env(env_name: str, budget: int = 50_000):
    if env_name == "swe_bench":
        return SWEBenchEnv(budget_tokens=budget, local_mode=True)
    if env_name == "livecodebench":
        return LiveCodeBenchEnv(budget_tokens=budget, local_mode=True)
    if env_name == "toolbench":
        return ToolBenchEnv(budget_tokens=budget)
    if env_name == "tau_bench":
        return TauBenchEnv(budget_tokens=budget)
    raise ValueError(f"Unknown env: {env_name}")
