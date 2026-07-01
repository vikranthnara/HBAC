from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"  # prerequisite not met (e.g. dataset volume)
    WARN = "warn"


@dataclass
class GateResult:
    gate_id: str
    phase: str
    name: str
    status: GateStatus
    measured: float | int | str | None
    threshold: float | int | str | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "phase": self.phase,
            "name": self.name,
            "status": self.status.value,
            "measured": self.measured,
            "threshold": self.threshold,
            "detail": self.detail,
        }


@dataclass
class GateReport:
    phase1_ready: bool
    phase2_ready: bool
    phase3_gateway_ready: bool
    results: list[GateResult] = field(default_factory=list)

    @property
    def go_phase3(self) -> bool:
        return (
            self.phase1_ready
            and self.phase2_ready
            and self.phase3_gateway_ready
            and all(r.status == GateStatus.PASS for r in self.results)
        )

    @property
    def all_pass(self) -> bool:
        return all(r.status == GateStatus.PASS for r in self.results)

    def add(self, result: GateResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict:
        return {
            "phase1_ready": self.phase1_ready,
            "phase2_ready": self.phase2_ready,
            "phase3_gateway_ready": self.phase3_gateway_ready,
            "go_phase3": self.go_phase3,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def summary_lines(self) -> list[str]:
        lines = [
            f"Phase 1 ready: {self.phase1_ready}",
            f"Phase 2 ready: {self.phase2_ready}",
            f"Phase 3 gateway ready: {self.phase3_gateway_ready}",
            f"GO Phase 3: {self.go_phase3}",
            "",
        ]
        for r in self.results:
            icon = {"pass": "PASS", "fail": "FAIL", "blocked": "BLOCKED", "warn": "WARN"}[r.status.value]
            lines.append(f"[{icon}] {r.phase}/{r.gate_id}: {r.detail}")
        return lines
