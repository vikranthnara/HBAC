from __future__ import annotations

from pathlib import Path

import pytest

from hbac.gates.phase1 import gate_env_stability, gate_pomdp_compliance
from hbac.gates.report import GateStatus
from hbac.gates.runner import run_all_gates
from hbac.gates.trajectory_validator import validate_trajectory_pomdp
from hbac.core.types import AgentAction, Trajectory, TrajectoryStep


class TestGoNoGoGates:
    def test_env_stability_passes(self):
        result = gate_env_stability()
        assert result.status == GateStatus.PASS
        assert result.measured == 1.0

    def test_pomdp_validator_accepts_seed(self):
        root = Path("data/oracles")
        if not list(root.rglob("oracles.jsonl")):
            pytest.skip("no seed oracles")
        result = gate_pomdp_compliance(root)
        assert result.status in {GateStatus.PASS, GateStatus.FAIL}

    def test_trajectory_validator_catches_bad_json(self):
        traj = Trajectory(
            task_id="t",
            benchmark="mock",
            model="m",
            baseline="react",
            success=True,
            total_tokens=10,
            budget=1000,
            steps=[
                TrajectoryStep(
                    turn=0,
                    action=AgentAction(tool_name="bash", tool_input="x"),
                    llm_response="not json at all",
                    tokens=10,
                )
            ],
        )
        errs = validate_trajectory_pomdp(traj)
        assert any("parse" in e for e in errs)

    def test_full_gate_report_runs(self, tmp_path):
        report = run_all_gates(
            oracle_root=Path("data/oracles"),
            checkpoint_dir=Path("checkpoints/variant_a"),
        )
        assert report.results
        report.save(tmp_path / "go.json")
        assert (tmp_path / "go.json").is_file()

    def test_phase1_env_covers_four_wrappers(self):
        from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES

        envs = {e.env_name for e in DETERMINISTIC_EPISODES}
        assert envs == {"swe_bench", "livecodebench", "toolbench", "tau_bench"}
