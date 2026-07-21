"""Tests for Phase 3c capability / DPO pair construction."""

from pathlib import Path

from hbac.training.capability import analyze_capability_deficits, build_dpo_pairs


def test_build_dpo_pairs_from_seed_oracles():
    root = Path("data/oracles/seed")
    runs = list(root.glob("*/oracles.jsonl"))
    if not runs:
        return
    pairs = build_dpo_pairs(root, limit=20, reject_modes=("wrong_tool",))
    assert pairs, "expected DPO pairs from seed oracles"
    assert pairs[0].chosen_reward >= pairs[0].rejected_reward
    assert pairs[0].capability_id == "wrong_tool"
    margins = [p.chosen_reward - p.rejected_reward for p in pairs]
    assert margins == sorted(margins, reverse=True)


def test_analyze_deficits_runs():
    root = Path("data/oracles")
    if not any(root.rglob("oracles.jsonl")):
        return
    deficits = analyze_capability_deficits(root)
    assert isinstance(deficits, list)
