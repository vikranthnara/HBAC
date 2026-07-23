"""Tests for local SWE gold-patch seeding and grading."""

from __future__ import annotations

from hbac.core.types import AgentAction
from hbac.envs.swe_bench import SWEBenchEnv
from hbac.envs.swe_local import (
    grade_workspace_against_gold,
    parse_unified_diff,
    seed_workspace_from_gold,
)


GOLD = """diff --git a/pkg/util.py b/pkg/util.py
--- a/pkg/util.py
+++ b/pkg/util.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def test_parse_unified_diff_before_after() -> None:
    parsed = parse_unified_diff(GOLD)
    assert "pkg/util.py" in parsed.files
    assert "return a - b" in parsed.files["pkg/util.py"].before
    assert "return a + b" in parsed.files["pkg/util.py"].after


def test_seed_and_grade_gold_match(tmp_path) -> None:
    ws = tmp_path / "repo"
    seed_workspace_from_gold(ws, GOLD)
    assert (ws / "pkg" / "util.py").is_file()
    assert "return a - b" in (ws / "pkg" / "util.py").read_text()

    ok, msg = grade_workspace_against_gold(ws, GOLD)
    assert not ok
    assert "mismatch" in msg or "content" in msg

    (ws / "pkg" / "util.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    ok, msg = grade_workspace_against_gold(ws, GOLD)
    assert ok, msg


def test_swe_local_micro_solvable() -> None:
    env = SWEBenchEnv(local_mode=True, budget_tokens=2000)
    obs = env.reset("swe-local-1")
    obs_s = str(obs)
    assert "foo.py" in obs_s or "Seeded" in obs_s or (env._workspace / "foo.py").is_file()
    assert (env._workspace / "foo.py").is_file()

    # Wrong fix → fail
    env.step(
        AgentAction(
            tool_name="str_replace_editor",
            tool_input={"path": "foo.py", "old_str": "return a - b", "new_str": "return a * b"},
        )
    )
    env.step(AgentAction(tool_name="submit", tool_input=""))
    assert env.evaluate().success is False

    env = SWEBenchEnv(local_mode=True, budget_tokens=2000)
    env.reset("swe-local-1")
    env.step(
        AgentAction(
            tool_name="str_replace_editor",
            tool_input={"path": "foo.py", "old_str": "return a - b", "new_str": "return a + b"},
        )
    )
    env.step(AgentAction(tool_name="submit", tool_input=""))
    result = env.evaluate()
    assert result.success is True, result.test_output


def test_fuzzy_grade_accepts_gold_lines(tmp_path) -> None:
    from hbac.envs.swe_local import grade_workspace_fuzzy, seed_workspace_from_gold

    ws = tmp_path / "repo"
    seed_workspace_from_gold(ws, GOLD)
    # Wrong exact content but includes the gold added line
    (ws / "pkg" / "util.py").write_text(
        "def add(a, b):\n    # note\n    return a + b\n", encoding="utf-8"
    )
    ok, msg = grade_workspace_fuzzy(ws, GOLD)
    assert ok, msg


def test_swe_gold_instance_solvable() -> None:
    env = SWEBenchEnv(local_mode=True, budget_tokens=2000)
    env._instances["demo__bug-1"] = {
        "instance_id": "demo__bug-1",
        "repo": "demo/repo",
        "base_commit": "abc",
        "problem_statement": "Fix add() in pkg/util.py",
        "patch": GOLD,
    }
    env.reset("demo__bug-1")
    assert (env._workspace / "pkg" / "util.py").is_file()
    env.step(
        AgentAction(
            tool_name="str_replace_editor",
            tool_input={
                "path": "pkg/util.py",
                "old_str": "return a - b",
                "new_str": "return a + b",
            },
        )
    )
    env.step(AgentAction(tool_name="submit", tool_input=""))
    result = env.evaluate()
    assert result.success is True, result.test_output
