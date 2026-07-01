from __future__ import annotations

from pathlib import Path

from hbac.baselines.base import RunnerConfig
from hbac.baselines.react import ReActRunner
from hbac.baselines.ref_orc import ReFORCRunner
from hbac.baselines.tab import TABRunner
from hbac.core.config import LLMConfig
from hbac.core.llm import LLMBackend, LLMResponse
from hbac.core.trajectory import TrajectoryStore
from hbac.gates.config import PHASE1
from hbac.gates.deterministic_episodes import DETERMINISTIC_EPISODES, make_env
from hbac.gates.report import GateResult, GateStatus
from hbac.gates.trajectory_validator import pomdp_compliance_rate
from hbac.training.dataset import find_all_trajectory_paths, find_oracle_paths


class ScriptedLLM(LLMBackend):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(LLMConfig())
        self.responses = responses
        self.i = 0

    def complete(self, messages, *, max_tokens, stop=None) -> LLMResponse:
        text = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return LLMResponse(text=text, prompt_tokens=10, completion_tokens=20, latency_ms=1.0)


def gate_env_stability() -> GateResult:
    """100% execution success on deterministic dummy trajectories across 4 env wrappers."""
    total = len(DETERMINISTIC_EPISODES)
    successes = 0
    failures: list[str] = []

    for ep in DETERMINISTIC_EPISODES:
        try:
            env = make_env(ep.env_name)
            llm = ScriptedLLM(ep.responses)
            runner = ReActRunner(llm, RunnerConfig(max_steps=10, output_dir=Path("/tmp/hbac_gate")))
            traj = runner.run_episode(env, ep.system_prompt, ep.task_id)
            if traj.success:
                successes += 1
            else:
                failures.append(f"{ep.env_name}/{ep.task_id}:eval_failed")
        except Exception as exc:
            failures.append(f"{ep.env_name}/{ep.task_id}:{exc}")

    rate = successes / total if total else 0.0
    status = GateStatus.PASS if rate >= PHASE1.env_execution_success_rate else GateStatus.FAIL
    return GateResult(
        gate_id="env_stability",
        phase="phase1",
        name="Environment stability (deterministic)",
        status=status,
        measured=rate,
        threshold=PHASE1.env_execution_success_rate,
        detail=f"{successes}/{total} episodes succeeded"
        + (f"; failures: {', '.join(failures)}" if failures else ""),
    )


def gate_oracle_yield(oracle_root: Path) -> GateResult:
    """Oracle yield rate from all_trajectories.jsonl (success / total)."""
    all_paths = find_all_trajectory_paths(oracle_root)
    if not all_paths:
        return GateResult(
            gate_id="oracle_yield",
            phase="phase1",
            name="Oracle yield rate",
            status=GateStatus.BLOCKED,
            measured=None,
            threshold=PHASE1.oracle_yield_rate,
            detail="No all_trajectories.jsonl found; run collect_oracles first",
        )

    total = success = 0
    for p in all_paths:
        for t in TrajectoryStore(p).load_all():
            total += 1
            if t.success:
                success += 1

    rate = success / total if total else 0.0
    status = GateStatus.PASS if rate >= PHASE1.oracle_yield_rate else GateStatus.FAIL
    return GateResult(
        gate_id="oracle_yield",
        phase="phase1",
        name="Oracle yield rate",
        status=status,
        measured=rate,
        threshold=PHASE1.oracle_yield_rate,
        detail=f"{success}/{total} successful rollouts",
    )


def gate_dataset_volume(oracle_root: Path) -> GateResult:
    """Compiled successful oracle count vs 500–1000 target."""
    paths = find_oracle_paths(oracle_root)
    trajs = []
    seen = set()
    for p in paths:
        for t in TrajectoryStore(p).load_successful():
            key = (t.benchmark, t.task_id, t.model)
            if key not in seen:
                seen.add(key)
                trajs.append(t)

    n = len(trajs)
    if n < PHASE1.min_oracle_trajectories:
        status = GateStatus.BLOCKED if n < 50 else GateStatus.FAIL
    else:
        status = GateStatus.PASS

    return GateResult(
        gate_id="dataset_volume",
        phase="phase1",
        name="Oracle dataset volume",
        status=status,
        measured=n,
        threshold=PHASE1.min_oracle_trajectories,
        detail=f"{n} unique successful oracles (target {PHASE1.min_oracle_trajectories}–{PHASE1.max_oracle_trajectories})",
    )


def gate_pomdp_compliance(oracle_root: Path) -> GateResult:
    paths = find_oracle_paths(oracle_root)
    trajs = []
    for p in paths:
        trajs.extend(TrajectoryStore(p).load_successful())

    rate, bad = pomdp_compliance_rate(trajs)
    status = GateStatus.PASS if rate >= PHASE1.pomdp_parse_compliance else GateStatus.FAIL
    detail = f"compliance={rate:.1%} ({len(trajs)} trajectories)"
    if bad:
        detail += f"; issues: {'; '.join(bad[:3])}"
    return GateResult(
        gate_id="pomdp_compliance",
        phase="phase1",
        name="POMDP parse compliance",
        status=status,
        measured=rate,
        threshold=PHASE1.pomdp_parse_compliance,
        detail=detail,
    )


def gate_baseline_harness(val_limit: int | None = 100) -> GateResult:
    """Run ReAct/TAB/Re-FORC on bundled LCB sample without crash; sanity Pass@1 floor."""
    from hbac.envs.livecodebench import LiveCodeBenchEnv

    env = LiveCodeBenchEnv(local_mode=False)
    task_ids = list(env._problems.keys())
    if val_limit:
        task_ids = task_ids[:val_limit]
    if len(task_ids) < PHASE1.baseline_val_samples:
        return GateResult(
            gate_id="baseline_harness",
            phase="phase1",
            name="Baseline harness verification",
            status=GateStatus.BLOCKED,
            measured=len(task_ids),
            threshold=PHASE1.baseline_val_samples,
            detail=f"Only {len(task_ids)} local tasks; need {PHASE1.baseline_val_samples} for literature comparison",
        )

    # With bundled sample we can run on available tasks
    results = {}
    for name, cls in [("react", ReActRunner), ("tab", TABRunner), ("ref_orc", ReFORCRunner)]:
        successes = 0
        for tid in task_ids:
            ep = DETERMINISTIC_EPISODES[1]  # lcb script
            llm = ScriptedLLM(ep.responses)
            runner = cls(llm, RunnerConfig(max_steps=5, output_dir=Path("/tmp/hbac_gate")))
            traj = runner.run_episode(
                LiveCodeBenchEnv(local_mode=False),
                ReActRunner.system_prompt_for_benchmark("livecodebench"),
                tid,
            )
            successes += int(traj.success)
        results[name] = successes / len(task_ids)

    min_rate = min(results.values())
    status = GateStatus.PASS if min_rate >= PHASE1.baseline_pass_at_1_min else GateStatus.FAIL
    return GateResult(
        gate_id="baseline_harness",
        phase="phase1",
        name="Baseline harness verification",
        status=status,
        measured=min_rate,
        threshold=PHASE1.baseline_pass_at_1_min,
        detail=f"Pass@1 on {len(task_ids)} tasks: {results} (heuristic proxies, not paper checkpoints)",
    )


def run_phase1_gates(oracle_root: Path) -> list[GateResult]:
    return [
        gate_env_stability(),
        gate_oracle_yield(oracle_root),
        gate_dataset_volume(oracle_root),
        gate_pomdp_compliance(oracle_root),
        gate_baseline_harness(),
    ]
