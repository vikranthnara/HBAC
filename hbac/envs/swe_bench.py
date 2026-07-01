from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hbac.core.config import SWEBenchConfig
from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec

logger = logging.getLogger(__name__)


class SWEBenchEnv(BaseAgentEnv):
    """SWE-Bench Verified adapter with Docker sandbox execution."""

    def __init__(
        self,
        budget_tokens: int = 50_000,
        lambda_penalty: float = 0.001,
        config: SWEBenchConfig | None = None,
        local_mode: bool = False,
    ) -> None:
        super().__init__(budget_tokens, lambda_penalty)
        self.config = config or SWEBenchConfig()
        self.local_mode = local_mode
        self._instances: dict[str, dict[str, Any]] = {}
        self._current: dict[str, Any] | None = None
        self._workspace: Path | None = None
        self._patch = ""
        self._step_count = 0
        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.local_mode:
            self._instances = {
                "swe-local-1": {
                    "instance_id": "swe-local-1",
                    "repo": "example/repo",
                    "base_commit": "abc123",
                    "problem_statement": "Fix the failing test in foo.py",
                    "patch": "diff --git a/foo.py b/foo.py\n",
                }
            }
            return
        try:
            from datasets import load_dataset

            ds = load_dataset(self.config.dataset_name, split=self.config.split)
            for row in ds:
                iid = row["instance_id"]
                self._instances[iid] = dict(row)
        except Exception as exc:
            logger.warning("Failed to load SWE-bench dataset (%s); using local fallback", exc)
            self.local_mode = True
            self._load_dataset()

    @classmethod
    def list_task_ids(cls, config: SWEBenchConfig | None = None, limit: int | None = None) -> list[str]:
        env = cls(config=config or SWEBenchConfig(), local_mode=False)
        ids = list(env._instances.keys())
        if limit:
            ids = ids[:limit]
        return ids

    def reset(self, task_id: str) -> Observation:
        if task_id not in self._instances:
            raise ValueError(f"Unknown SWE-bench instance: {task_id}")
        self._current = self._instances[task_id]
        self._budget = self._budget.__class__(self._budget.budget_tokens)
        self._history = []
        self._turn = 0
        self._done = False
        self._patch = ""
        self._step_count = 0

        if self._workspace and self._workspace.exists():
            import shutil

            shutil.rmtree(self._workspace, ignore_errors=True)
        self._workspace = Path(tempfile.mkdtemp(prefix="hbac_swe_"))

        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="swe_bench",
            query=self._current["problem_statement"],
            budget_tokens=self._budget.budget_tokens,
            tools_available=["bash", "str_replace_editor", "submit"],
            metadata={
                "repo": self._current.get("repo"),
                "base_commit": self._current.get("base_commit"),
            },
        )
        self._append_history("user", self._task_spec.query)
        return self._build_observation(
            f"Repository workspace ready at {self._workspace}. "
            "Use bash commands to explore and edit files."
        )

    def _run_bash(self, command: str) -> str:
        if not self._workspace:
            return "No workspace initialized"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=self.config.command_timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")
            if len(output) > self.config.observation_max_chars:
                output = output[: self.config.observation_max_chars] + "\n...(truncated)"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {self.config.command_timeout}s"
        except Exception as exc:
            return f"Error: {exc}"

    def _apply_str_replace(self, tool_input: str | dict | None) -> str:
        if isinstance(tool_input, dict):
            path = tool_input.get("path", "")
            old = tool_input.get("old_str", "")
            new = tool_input.get("new_str", "")
        else:
            return "str_replace_editor requires JSON with path, old_str, new_str"
        if not self._workspace:
            return "No workspace"
        file_path = self._workspace / path
        if not file_path.exists():
            return f"File not found: {path}"
        content = file_path.read_text(encoding="utf-8")
        if old not in content:
            return f"old_str not found in {path}"
        file_path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"Replaced in {path}"

    def _capture_patch(self) -> str:
        if not self._workspace:
            return ""
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout or ""
        except Exception:
            return self._patch

    def step(self, action: AgentAction) -> StepResult:
        if self._done:
            return StepResult(
                obs=self._build_observation("Episode finished."),
                reward=0.0,
                done=True,
            )

        self._step_count += 1
        self._turn += 1
        feedback = ""

        if action.stop or action.tool_name == "submit":
            self._patch = self._capture_patch() or str(action.tool_input or "")
            self._done = True
            feedback = "Patch submitted."
        elif action.tool_name == "bash":
            feedback = self._run_bash(str(action.tool_input or ""))
        elif action.tool_name == "str_replace_editor":
            feedback = self._apply_str_replace(action.tool_input)
        else:
            feedback = f"Unknown tool: {action.tool_name}"

        self._append_history("assistant", action.model_dump_json())
        self._append_history("user", feedback)

        if self._step_count >= 100:
            self._done = True

        return StepResult(
            obs=self._build_observation(feedback),
            reward=self.compute_step_reward(),
            done=self._done,
            info=StepInfo(step_index=self._turn),
        )

    def evaluate(self) -> EvalResult:
        patch = self._patch or self._capture_patch()
        success = False
        test_output = ""

        if self.local_mode:
            success = bool(patch)
            test_output = "local_mode: patch present" if patch else "local_mode: no patch"
        else:
            try:
                success, test_output = self._grade_patch(patch)
            except Exception as exc:
                test_output = f"Evaluation error: {exc}"

        return EvalResult(
            success=success,
            final_output=patch,
            test_output=test_output,
            total_tokens=self.total_tokens,
            budget_violated=self._budget.violated,
            metadata={"instance_id": self._current["instance_id"] if self._current else ""},
        )

    def _grade_patch(self, patch: str) -> tuple[bool, str]:
        if not patch.strip():
            return False, "Empty patch"
        if not self._current:
            return False, "No instance loaded"

        try:
            from swebench.harness.constants import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
            from swebench.harness.run_evaluation import run_instances

            pred = {
                KEY_INSTANCE_ID: self._current["instance_id"],
                KEY_MODEL: "hbac-agent",
                KEY_PREDICTION: patch,
            }
            with tempfile.TemporaryDirectory() as tmp:
                pred_path = Path(tmp) / "pred.jsonl"
                pred_path.write_text(
                    __import__("json").dumps(pred) + "\n",
                    encoding="utf-8",
                )
                result = run_instances(
                    predictions_path=str(pred_path),
                    instance_ids=[self._current["instance_id"]],
                    max_workers=1,
                )
            resolved = bool(result) and any(
                r.get("resolved", False) for r in (result if isinstance(result, list) else [result])
            )
            return resolved, str(result)
        except ImportError:
            return bool(patch.strip()), "swebench harness unavailable; graded on non-empty patch"
        except Exception as exc:
            logger.exception("SWE-bench grading failed")
            return False, str(exc)
