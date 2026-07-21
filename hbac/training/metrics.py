"""Shared batch metrics: compliant utility, token-normalized success."""

from __future__ import annotations


def compliant_batch_utility(
    mean_batch_reward: float,
    batch_violation_rate: float,
    *,
    pass_at_1: float | None = None,
    parse_failure_rate: float = 0.0,
    parse_penalty: float = 0.5,
) -> float:
    """
    Utility that penalizes budget violations and parse failures.

    U = R * (1 - violation_rate) - parse_penalty * parse_failures
    Optional pass@1 blend when reward is uninformative (e.g. oracle).
    """
    base = mean_batch_reward * (1.0 - batch_violation_rate)
    base -= parse_penalty * parse_failure_rate
    if pass_at_1 is not None and mean_batch_reward <= 0:
        base = pass_at_1 * (1.0 - batch_violation_rate) - parse_penalty * parse_failure_rate
    return base


def reward_per_success(
    pass_at_1: float,
    mean_tokens_used: float,
    *,
    eps: float = 1e-6,
) -> float:
    """Success rate normalized by token cost (higher = better efficiency)."""
    if mean_tokens_used <= 0:
        return 0.0
    return pass_at_1 / (mean_tokens_used + eps)


def summarize_allocator_row(row: dict) -> dict:
    """Derive P2/P3 metrics from a live or oracle compose JSON row."""
    viol = float(row.get("batch_violation_rate") or 0)
    rew = float(row.get("mean_batch_reward") or 0)
    p = float(row.get("pass_at_1") or 0)
    tok = float(row.get("mean_tokens_used") or 0)
    parse_f = float(row.get("mean_parse_failures_per_task") or 0)
    return {
        "pass_at_1": p,
        "mean_batch_reward": rew,
        "mean_tokens_used": tok,
        "batch_violation_rate": viol,
        "mean_parse_failures_per_task": parse_f,
        "compliant_utility": compliant_batch_utility(rew, viol, pass_at_1=p, parse_failure_rate=parse_f),
        "reward_per_success_per_token": reward_per_success(p, tok),
    }
