from __future__ import annotations

from pathlib import Path

from hbac.gates.phase1 import run_phase1_gates
from hbac.gates.phase2 import run_phase2_gates
from hbac.gates.phase3_gateway import run_phase3_gateway_gates
from hbac.gates.report import GateReport, GateStatus


def _phase_ready(results: list, phase: str) -> bool:
    phase_results = [r for r in results if r.phase == phase]
    return all(r.status == GateStatus.PASS for r in phase_results)


def all_gates_pass(report) -> bool:
    return all(r.status == GateStatus.PASS for r in report.results)


def run_all_gates(
    oracle_root: Path | None = None,
    checkpoint_dir: Path | None = None,
) -> GateReport:
    oracle_root = oracle_root or Path("data/oracles")
    checkpoint_dir = checkpoint_dir or Path("checkpoints/variant_a")

    report = GateReport(phase1_ready=False, phase2_ready=False, phase3_gateway_ready=False)

    for r in run_phase1_gates(oracle_root):
        report.add(r)
    for r in run_phase2_gates(oracle_root, checkpoint_dir):
        report.add(r)
    for r in run_phase3_gateway_gates(oracle_root):
        report.add(r)

    report.phase1_ready = _phase_ready(report.results, "phase1")
    report.phase2_ready = _phase_ready(report.results, "phase2")
    report.phase3_gateway_ready = _phase_ready(report.results, "phase3_gateway")
    return report
