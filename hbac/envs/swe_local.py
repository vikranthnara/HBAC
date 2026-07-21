"""Local SWE harness helpers: seed a git workspace from a gold patch and grade against it.

Live eval historically used an empty tempfile + ``success = bool(patch)``, which
made SWE pass@1 structurally 0% even for capable models. This module reconstructs
pre-patch files from the gold unified diff, commits them, and grades by matching
post-patch file contents after the agent edits.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


_HUNK_RE = re.compile(r"^@@")
_DIFF_GIT_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")


@dataclass
class FileEdit:
    path: str
    before: str = ""
    after: str = ""
    is_new: bool = False
    is_deleted: bool = False


@dataclass
class ParsedPatch:
    files: dict[str, FileEdit] = field(default_factory=dict)

    @property
    def touched_paths(self) -> list[str]:
        return list(self.files.keys())


def parse_unified_diff(patch: str) -> ParsedPatch:
    """Parse a SWE-bench-style unified diff into before/after file bodies."""
    result = ParsedPatch()
    if not patch or not patch.strip():
        return result

    lines = patch.splitlines(keepends=True)
    i = 0
    current: FileEdit | None = None
    before_buf: list[str] = []
    after_buf: list[str] = []

    def _flush() -> None:
        nonlocal current, before_buf, after_buf
        if current is None:
            return
        current.before = "".join(before_buf)
        current.after = "".join(after_buf)
        result.files[current.path] = current
        current = None
        before_buf, after_buf = [], []

    while i < len(lines):
        line = lines[i]
        m = _DIFF_GIT_RE.match(line.rstrip("\n"))
        if m:
            _flush()
            path_b = m.group(2).strip()
            current = FileEdit(path=path_b)
            before_buf, after_buf = [], []
            i += 1
            continue

        if line.startswith("--- "):
            raw = line[4:].strip()
            if raw.startswith("a/"):
                raw = raw[2:]
            if raw == "/dev/null" and current is not None:
                current.is_new = True
            i += 1
            continue

        if line.startswith("+++ "):
            raw = line[4:].strip()
            if raw.startswith("b/"):
                raw = raw[2:]
            if current is not None:
                if raw == "/dev/null":
                    current.is_deleted = True
                elif raw:
                    current.path = raw
            i += 1
            continue

        if _HUNK_RE.match(line):
            i += 1
            while i < len(lines):
                hl = lines[i]
                if (
                    hl.startswith("diff --git ")
                    or hl.startswith("--- ")
                    or hl.startswith("+++ ")
                    or _HUNK_RE.match(hl)
                ):
                    break
                if hl.startswith("\\"):  # "\ No newline at end of file"
                    i += 1
                    continue
                if not hl:
                    # Empty line inside hunk — treat as context newline if present
                    before_buf.append("\n")
                    after_buf.append("\n")
                    i += 1
                    continue
                tag, body = hl[0], hl[1:]
                if tag == " ":
                    before_buf.append(body)
                    after_buf.append(body)
                elif tag == "-":
                    before_buf.append(body)
                elif tag == "+":
                    after_buf.append(body)
                else:
                    # Malformed / binary marker — stop hunk
                    break
                i += 1
            continue

        i += 1

    _flush()
    return result


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


def init_git_repo(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / ".git").exists():
        _run_git(workspace, "init")
        _run_git(workspace, "config", "user.email", "hbac@local")
        _run_git(workspace, "config", "user.name", "hbac")


def seed_workspace_from_gold(workspace: Path, gold_patch: str) -> ParsedPatch:
    """Write pre-patch files from gold diff and create an initial git commit."""
    parsed = parse_unified_diff(gold_patch)
    init_git_repo(workspace)

    for path, edit in parsed.files.items():
        if edit.is_new:
            # New file in gold → absent in before state
            continue
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.before, encoding="utf-8")

    _run_git(workspace, "add", "-A")
    # Allow empty commit if patch only adds files (rare)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "hbac swe local seed"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return parsed


def seed_micro_task(workspace: Path, task_id: str = "swe-local-1") -> ParsedPatch:
    """Deterministic solvable bug when no gold patch is available."""
    init_git_repo(workspace)
    foo = workspace / "foo.py"
    test = workspace / "test_foo.py"
    foo.write_text(
        "def add(a: int, b: int) -> int:\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a - b\n",
        encoding="utf-8",
    )
    test.write_text(
        "from foo import add\n\n\ndef test_add() -> None:\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )
    _run_git(workspace, "add", "-A")
    _run_git(workspace, "commit", "-m", f"seed {task_id}")
    gold = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a: int, b: int) -> int:\n"
        '     """Return the sum of a and b."""\n'
        "-    return a - b\n"
        "+    return a + b\n"
    )
    return parse_unified_diff(gold)


def grade_workspace_against_gold(workspace: Path, gold_patch: str) -> tuple[bool, str]:
    """Success iff every gold after-path matches workspace file contents."""
    parsed = parse_unified_diff(gold_patch)
    if not parsed.files:
        return False, "local_grade: empty or unparseable gold patch"

    mismatches: list[str] = []
    for path, edit in parsed.files.items():
        target = workspace / path
        if edit.is_deleted:
            if target.exists():
                mismatches.append(f"{path}: expected deleted")
            continue
        if not target.exists():
            mismatches.append(f"{path}: missing")
            continue
        got = target.read_text(encoding="utf-8")
        if got != edit.after:
            mismatches.append(f"{path}: content mismatch")

    if mismatches:
        return False, "local_grade: " + "; ".join(mismatches[:5])
    return True, f"local_grade: matched {len(parsed.files)} file(s)"


def grade_micro_task(workspace: Path) -> tuple[bool, str]:
    """Run the seeded unit test for the micro fallback task."""
    try:
        proc = subprocess.run(
            ["python", "-c", "from foo import add; assert add(2, 3) == 5"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return False, f"local_grade_micro: {exc}"
    if proc.returncode == 0:
        return True, "local_grade_micro: tests passed"
    err = (proc.stderr or proc.stdout or "fail").strip()
    return False, f"local_grade_micro: {err[:500]}"
