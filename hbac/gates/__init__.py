"""Empirical Go/No-Go gates before Phase 3 full-scale training."""

from hbac.gates.report import GateReport, GateResult, GateStatus

__all__ = ["GateReport", "GateResult", "GateStatus", "run_all_gates"]


def run_all_gates(*args, **kwargs):
    from hbac.gates.runner import run_all_gates as _run

    return _run(*args, **kwargs)
