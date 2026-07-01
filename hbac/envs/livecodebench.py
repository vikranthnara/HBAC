from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from hbac.core.config import LiveCodeBenchConfig
from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec

logger = logging.getLogger(__name__)


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data.get("tool_input"), str):
                return data["tool_input"]
        except json.JSONDecodeError:
            pass
    return text.strip()


class LiveCodeBenchEnv(BaseAgentEnv):
    """Multi-turn codegeneration + self-repair wrapper for LiveCodeBench."""

    def __init__(
        self,
        budget_tokens: int = 50_000,
        lambda_penalty: float = 0.001,
        config: LiveCodeBenchConfig | None = None,
        local_mode: bool = False,
    ) -> None:
        super().__init__(budget_tokens, lambda_penalty)
        self.config = config or LiveCodeBenchConfig()
        self.local_mode = local_mode
        self._problems: dict[str, dict[str, Any]] = {}
        self._current: dict[str, Any] | None = None
        self._code = ""
        self._repair_turn = 0
        self._last_test_feedback = ""
        self._load_problems()

    def _load_problems(self) -> None:
        if self.local_mode:
            self._problems = {
                "lcb-local-1": {
                    "question_id": "lcb-local-1",
                    "question_content": "Write a function add(a, b) that returns a + b.",
                    "starter_code": "",
                    "public_test_cases": json.dumps(
                        [{"input": "1\n2", "output": "3", "testtype": "stdin"}]
                    ),
                    "metadata": json.dumps({}),
                }
            }
            return

        try:
            problems = self._load_from_lcb_runner()
            for p in problems:
                qid = str(p.get("question_id", p.get("instance_id", "")))
                self._problems[qid] = p
            if not self._problems:
                raise ValueError("No problems loaded")
        except Exception as exc:
            logger.warning("LiveCodeBench load failed (%s); using bundled sample data", exc)
            data_path = Path(__file__).parent / "data" / "lcb_sample.json"
            if data_path.exists():
                for p in json.loads(data_path.read_text(encoding="utf-8")):
                    self._problems[str(p["question_id"])] = p
            else:
                self.local_mode = True
                self._load_problems()

    def _load_from_lcb_runner(self) -> list[dict[str, Any]]:
        """Attempt to load problems via lcb_runner if installed."""
        try:
            from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

            benchmark = CodeGenerationProblem.load(self.config.release_version)
            return [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in benchmark]
        except ImportError:
            pass

        # Fallback: load bundled minimal dataset
        data_path = Path(__file__).parent / "data" / "lcb_sample.json"
        if data_path.exists():
            return json.loads(data_path.read_text(encoding="utf-8"))
        raise ImportError("lcb_runner not installed and no sample data found")

    @classmethod
    def list_task_ids(cls, config: LiveCodeBenchConfig | None = None, limit: int | None = None) -> list[str]:
        try:
            env = cls(config=config or LiveCodeBenchConfig(), local_mode=False)
            ids = list(env._problems.keys())
        except Exception:
            env = cls(local_mode=True)
            ids = list(env._problems.keys())
        if limit:
            ids = ids[:limit]
        return ids

    def reset(self, task_id: str) -> Observation:
        if task_id not in self._problems:
            raise ValueError(f"Unknown LiveCodeBench task: {task_id}")
        self._current = self._problems[task_id]
        self._budget = self._budget.__class__(self._budget.budget_tokens)
        self._history = []
        self._turn = 0
        self._done = False
        self._code = ""
        self._repair_turn = 0
        self._last_test_feedback = ""

        query = self._current.get("question_content", "")
        starter = self._current.get("starter_code", "")
        if starter:
            query += f"\n\nStarter code:\n```python\n{starter}\n```"

        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="livecodebench",
            query=query,
            budget_tokens=self._budget.budget_tokens,
            tools_available=["generate_code", "run_tests", "revise", "submit"],
            metadata={"question_id": task_id},
        )
        self._append_history("user", query)
        return self._build_observation(
            "Problem loaded. Use generate_code to write your solution, then run_tests."
        )

    def _run_tests_local(self, code: str) -> tuple[bool, str]:
        if not code.strip():
            return False, "No code to test"

        if self.local_mode:
            if "def add" in code or "+" in code:
                return True, "All tests passed (local mock)."
            return False, "Tests failed: expected addition logic."

        test_cases_raw = self._current.get("public_test_cases", "[]")
        test_cases = json.loads(test_cases_raw) if isinstance(test_cases_raw, str) else test_cases_raw

        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "solution.py"
            script.write_text(code, encoding="utf-8")
            failures = []
            for i, tc in enumerate(test_cases):
                inp = tc.get("input", "")
                expected = tc.get("output", "").strip()
                try:
                    proc = subprocess.run(
                        [sys.executable, str(script)],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=self.config.eval_timeout,
                    )
                    actual = (proc.stdout or "").strip()
                    if actual != expected:
                        failures.append(f"Test {i}: expected {expected!r}, got {actual!r}")
                except subprocess.TimeoutExpired:
                    failures.append(f"Test {i}: timeout")
                except Exception as exc:
                    failures.append(f"Test {i}: {exc}")

            if failures:
                return False, "Failures:\n" + "\n".join(failures)
            return True, "All public tests passed."

    def step(self, action: AgentAction) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._build_observation("Episode finished."),
                reward=0.0,
                done=True,
            )

        self._turn += 1
        feedback = ""
        tool = action.tool_name

        if action.stop or tool == "submit":
            self._done = True
            feedback = "Submitted final code."
        elif tool in {"generate_code", "revise"}:
            raw = action.tool_input if action.tool_input is not None else ""
            self._code = _extract_code(str(raw))
            self._repair_turn += 1 if tool == "revise" else 0
            feedback = f"Code updated ({len(self._code)} chars)."
            if tool == "generate_code":
                feedback += " Run run_tests to validate."
        elif tool == "run_tests":
            passed, feedback = self._run_tests_local(self._code)
            self._last_test_feedback = feedback
            if passed:
                self._done = True
                feedback += " You may submit."
        else:
            feedback = f"Unknown tool: {tool}. Use generate_code, run_tests, revise, or submit."

        if self._repair_turn >= self.config.max_repair_turns and not self._done:
            feedback += f"\nMax repair turns ({self.config.max_repair_turns}) reached."

        self._append_history("assistant", action.model_dump_json())
        self._append_history("user", feedback)

        return StepResult(
            obs=self._build_observation(feedback),
            reward=self.compute_step_reward(1.0 if "passed" in feedback.lower() else 0.0),
            done=self._done,
            info=StepInfo(
                step_index=self._turn,
                extra={"test_passed": "passed" in feedback.lower(), "repair_turn": self._repair_turn},
            ),
        )

    def evaluate(self) -> EvalResult:
        passed, test_output = self._run_tests_local(self._code)
        return EvalResult(
            success=passed,
            final_output=self._code,
            test_output=test_output,
            total_tokens=self.total_tokens,
            budget_violated=self._budget.violated,
            metadata={"repair_turns": self._repair_turn},
        )
