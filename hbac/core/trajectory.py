from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hbac.core.types import Trajectory


class TrajectoryStore:
    """JSONL persistence for trajectories."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trajectory.model_dump_json() + "\n")

    def load_all(self) -> list[Trajectory]:
        if not self.path.exists():
            return []
        trajectories: list[Trajectory] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(Trajectory.model_validate_json(line))
        return trajectories

    def load_successful(self) -> list[Trajectory]:
        return [t for t in self.load_all() if t.success]

    @staticmethod
    def write_many(path: Path, trajectories: Iterable[Trajectory]) -> None:
        store = TrajectoryStore(path)
        for traj in trajectories:
            store.append(traj)
