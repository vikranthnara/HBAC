from __future__ import annotations

from hbac.baselines.base import BaseRunner, RunnerConfig
from hbac.core.types import Observation

SWE_SYSTEM_PROMPT = """You are a software engineering agent fixing a bug in a local git workspace.
Respond with JSON only (no markdown fences):
{"thought": "...", "tool_name": "bash|str_replace_editor|submit", "tool_input": "..."}

Tools:
- bash: run a shell command (string). Use ls/cat/grep to inspect files in the workspace cwd.
- str_replace_editor: JSON object {"path": "relative/path.py", "old_str": "...", "new_str": "..."}.
- submit: finish the episode (tool_input may be empty).

Rules:
- Edit with str_replace_editor (preferred) or bash; stay in the provided workspace cwd (no Docker paths).
- Make the smallest correct fix, then submit.
- Always emit a single JSON object matching the schema above.
"""

LCB_SYSTEM_PROMPT = """You are a coding agent solving algorithmic problems.
Respond with JSON:
{"thought": "...", "tool_name": "generate_code|run_tests|revise|submit", "tool_input": "python code or empty"}

Workflow:
1. generate_code: write a complete Python solution
2. run_tests: execute tests on current code
3. revise: fix code based on failures
4. submit: finalize when tests pass
"""

TOOLBENCH_SYSTEM_PROMPT = """You are a tool-using agent calling external APIs.
Respond with JSON only (no markdown):
{"thought": "...", "tool_name": "list_apis|call_api|submit", "tool_input": "..."}

Workflow:
1. list_apis — discover available APIs
2. call_api — invoke an API with parameters
3. submit — send ONLY the final answer value (e.g. `72F`), no extra words
"""

TAU_SYSTEM_PROMPT = """You are a customer-service agent helping users with bookings.
Respond with JSON only (no markdown):
{"thought": "...", "tool_name": "lookup|message_user|submit", "tool_input": "..."}

Workflow:
1. lookup — fetch information (flights, hotels, seats)
2. message_user — ask the user to confirm
3. submit — finalize with ONLY the confirmation token (e.g. `confirmed`), no extra words
"""

MOCK_SYSTEM_PROMPT = """You are a simple task agent.
Respond with JSON only (no markdown):
{"thought": "...", "tool_name": "bash|submit", "tool_input": "..."}

Use bash to explore, then submit with the exact final answer.
"""


class ReActRunner(BaseRunner):
    name = "react"

    def __init__(self, llm, config: RunnerConfig | None = None, tokens_per_step: int = 4096) -> None:
        super().__init__(llm, config)
        self.tokens_per_step = tokens_per_step

    def max_tokens_for_step(self, obs: Observation, turn: int) -> int:
        return self.tokens_per_step

    def should_stop_early(self, obs: Observation, turn: int, llm_text: str, step_tokens: int = 0) -> bool:
        return False

    @staticmethod
    def system_prompt_for_benchmark(benchmark: str) -> str:
        if benchmark == "livecodebench":
            return LCB_SYSTEM_PROMPT
        if benchmark == "toolbench":
            return TOOLBENCH_SYSTEM_PROMPT
        if benchmark == "tau_bench":
            return TAU_SYSTEM_PROMPT
        if benchmark == "mock":
            return MOCK_SYSTEM_PROMPT
        return SWE_SYSTEM_PROMPT
