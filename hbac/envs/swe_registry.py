"""SWE-bench Lite instance registry for oracle replay (no Docker required)."""

from __future__ import annotations

import logging
from pathlib import Path

from hbac.envs.swe_bench import SWEBenchEnv
from hbac.training.dataset import find_oracle_paths
from hbac.core.trajectory import TrajectoryStore

logger = logging.getLogger(__name__)

_LITE_INSTANCES: dict[str, dict] | None = None


def _load_lite_instances() -> dict[str, dict]:
    global _LITE_INSTANCES
    if _LITE_INSTANCES is not None:
        return _LITE_INSTANCES

    instances: dict[str, dict] = {
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

    for path in find_oracle_paths(Path("data/oracles/swe_lite")):
        for traj in TrajectoryStore(path).load_successful():
            meta = traj.metadata or {}
            patch = meta.get("patch") or (traj.final_output if hasattr(traj, "final_output") else "")
            if not patch and traj.steps:
                last = traj.steps[-1].action
                if isinstance(last, dict):
                    patch = str(last.get("tool_input", ""))
            instances[traj.task_id] = {
                "instance_id": traj.task_id,
                "repo": meta.get("repo", "example/repo"),
                "base_commit": meta.get("base_commit", "abc"),
                "problem_statement": meta.get("problem_statement", f"Fix {traj.task_id}"),
                "patch": patch or "diff --git\n",
            }

    try:
        from hbac.core.config import SWEBenchConfig
        from hbac.envs.swe_bench import SWEBenchEnv as _Env

        cfg = SWEBenchConfig(dataset_name="princeton-nlp/SWE-bench_Lite", split="test")
        probe = _Env(config=cfg, local_mode=False, budget_tokens=50_000)
        for iid, row in probe._instances.items():
            if iid not in instances:
                instances[iid] = dict(row, instance_id=iid)
    except Exception as exc:
        logger.debug("SWE Lite HF preload skipped: %s", exc)

    _LITE_INSTANCES = instances
    return instances


def swe_env_for_task(task_id: str, budget: int) -> SWEBenchEnv:
    """Build a local-mode SWE env seeded from the Lite registry gold patch when available."""
    env = SWEBenchEnv(budget_tokens=budget, local_mode=True)
    registry = _load_lite_instances()
    if task_id in registry:
        env._instances[task_id] = registry[task_id]
    elif task_id not in env._instances:
        # Unknown ID: still allow reset via micro fallback by synthesizing an entry.
        env._instances[task_id] = {
            "instance_id": task_id,
            "repo": "local/micro",
            "base_commit": "local",
            "problem_statement": (
                f"Task {task_id}: foo.py has a bug in add(). "
                "Make add(2, 3) return 5, then submit."
            ),
            "patch": "",
        }
    return env


__all__ = ["swe_env_for_task", "_load_lite_instances"]
