from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hbac.core.config import SWEBenchConfig
from hbac.core.env import BaseAgentEnv
from hbac.core.types import AgentAction, EvalResult, Observation, StepInfo, StepResult, TaskSpec
from hbac.envs.swe_local import (
    grade_micro_task,
    grade_workspace_against_gold,
    grade_workspace_fuzzy,
    seed_micro_task,
    seed_workspace_from_gold,
)

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
        self._gold_patch = ""
        self._local_grade_mode = "gold"  # gold | micro
        self._load_dataset()

    def _load_dataset(self) -> None:
        if self.local_mode:
            self._instances = {
                "swe-local-1": {
                    "instance_id": "swe-local-1",
                    "repo": "example/repo",
                    "base_commit": "abc123",
                    "problem_statement": (
                        "foo.py has a bug: add(a, b) returns a - b instead of a + b. "
                        "Fix it so add(2, 3) == 5, then submit."
                    ),
                    "patch": "",
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
        self._gold_patch = str(self._current.get("patch") or "")
        self._local_grade_mode = "gold"

        if self.local_mode:
            seeded_paths: list[str] = []
            if self._gold_patch.strip():
                try:
                    parsed = seed_workspace_from_gold(self._workspace, self._gold_patch)
                    seeded_paths = parsed.touched_paths
                    if not seeded_paths and not any(
                        p for p in self._workspace.rglob("*") if p.is_file() and ".git" not in p.parts
                    ):
                        # Gold patch only adds files — still seed micro fallback
                        raise ValueError("gold patch produced empty before-state")
                except Exception as exc:
                    logger.warning("Gold seed failed (%s); using micro task", exc)
                    self._local_grade_mode = "micro"
                    parsed = seed_micro_task(self._workspace, task_id)
                    self._gold_patch = (
                        "diff --git a/foo.py b/foo.py\n"
                        "--- a/foo.py\n"
                        "+++ b/foo.py\n"
                        "@@ -1,3 +1,3 @@\n"
                        " def add(a: int, b: int) -> int:\n"
                        '     """Return the sum of a and b."""\n'
                        "-    return a - b\n"
                        "+    return a + b\n"
                    )
                    seeded_paths = parsed.touched_paths
            else:
                self._local_grade_mode = "micro"
                parsed = seed_micro_task(self._workspace, task_id)
                self._gold_patch = (
                    "diff --git a/foo.py b/foo.py\n"
                    "--- a/foo.py\n"
                    "+++ b/foo.py\n"
                    "@@ -1,3 +1,3 @@\n"
                    " def add(a: int, b: int) -> int:\n"
                    '     """Return the sum of a and b."""\n'
                    "-    return a - b\n"
                    "+    return a + b\n"
                )
                seeded_paths = parsed.touched_paths
            file_list = ", ".join(seeded_paths[:12]) or "foo.py"
            snippets: list[str] = []
            for rel in seeded_paths[:6]:
                fp = self._workspace / rel
                if not fp.is_file():
                    continue
                body = fp.read_text(encoding="utf-8")
                if len(body) > 2500:
                    body = body[:2500] + "\n...(truncated)"
                snippets.append(f"----- {rel} -----\n{body}")
            issue = str(self._current.get("problem_statement") or "").strip()
            if len(issue) > 1200:
                issue = issue[:1200] + "\n...(truncated)"
            local_query = (
                f"Local SWE task `{task_id}`.\n"
                f"Workspace cwd is already set; edit relative paths only.\n"
                f"Issue (may be truncated):\n{issue or '(see seeded files)'}\n\n"
                f"Seeded files ({file_list}):\n"
                + ("\n\n".join(snippets) if snippets else "(see workspace)")
                + "\n\nFix the bug with str_replace_editor, then submit."
            )
            boot = (
                f"Workspace ready at {self._workspace}. "
                f"Seeded: {file_list}. Tools: bash, str_replace_editor, submit."
            )
        else:
            local_query = str(self._current["problem_statement"])
            boot = (
                f"Repository workspace ready at {self._workspace}. "
                "Use bash commands to explore and edit files."
            )

        self._task_spec = TaskSpec(
            task_id=task_id,
            benchmark="swe_bench",
            query=local_query,
            budget_tokens=self._budget.budget_tokens,
            tools_available=["bash", "str_replace_editor", "submit"],
            metadata={
                "repo": self._current.get("repo"),
                "base_commit": self._current.get("base_commit"),
                "local_grade_mode": self._local_grade_mode if self.local_mode else "docker",
            },
        )
        self._append_history("user", self._task_spec.query)
        return self._build_observation(boot)

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
            # Include both tracked diffs and unstaged new files after seed commit.
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            diff = result.stdout or ""
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if untracked.stdout.strip():
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=self._workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                result = subprocess.run(
                    ["git", "diff", "--cached", "HEAD"],
                    cwd=self._workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                diff = result.stdout or diff
            return diff
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
            if self._workspace is None:
                success, test_output = False, "local_mode: no workspace"
            else:
                mode = (
                    os.environ.get("HBAC_SWE_LOCAL_GRADE", self._local_grade_mode or "gold")
                    .strip()
                    .lower()
                )
                if mode in {"micro"} or self._local_grade_mode == "micro":
                    success, test_output = grade_micro_task(self._workspace)
                elif mode in {"fuzzy", "touched"}:
                    if self._gold_patch.strip():
                        success, test_output = grade_workspace_fuzzy(
                            self._workspace, self._gold_patch
                        )
                    else:
                        success, test_output = grade_micro_task(self._workspace)
                elif self._gold_patch.strip():
                    success, test_output = grade_workspace_against_gold(
                        self._workspace, self._gold_patch
                    )
                else:
                    success, test_output = bool(patch.strip()), (
                        "local_mode: patch present (no gold)"
                        if patch.strip()
                        else "local_mode: no patch"
                    )
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
