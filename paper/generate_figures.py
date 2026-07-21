"""Generate camera-ready figures for HBAC paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
    }
)


def floor_dose_response() -> None:
    data = json.loads((ROOT.parent / "results/fair_floor_sweep_analysis.json").read_text())
    floors = [r["floor"] for r in data["rows"]]
    fair = [r["hbac_fair_pass_at_1"] * 100 for r in data["rows"]]
    prior = [r["type_prior_pass_at_1"] * 100 for r in data["rows"]]
    fair_lo = [r["hbac_fair_ci95"][0] * 100 for r in data["rows"]]
    fair_hi = [r["hbac_fair_ci95"][1] * 100 for r in data["rows"]]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(floors))
    ax.fill_between(x, fair_lo, fair_hi, alpha=0.2, color="#2166ac")
    ax.plot(x, fair, "o-", color="#2166ac", label="HBAC fair", linewidth=1.8, markersize=5)
    ax.plot(x, prior, "s--", color="#b2182b", label="Type-prior", linewidth=1.8, markersize=5)
    ax.set_xticks(x, [str(f) for f in floors])
    ax.set_xlabel("Per-task token floor")
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Floor dose-response (V3 pool, $n{=}300$/floor)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_ylim(20, 35)
    fig.tight_layout()
    fig.savefig(FIG / "floor_dose_response.pdf", bbox_inches="tight")
    plt.close(fig)


def per_benchmark() -> None:
    live_path = ROOT.parent / "results/rivanna/compose_live_v3_d18_floor400_n2000.json"
    if not live_path.is_file():
        live_path = ROOT.parent / "results/rivanna/compose_live_v3_heuristics_floor400_n1000_dpo_v2.json"
    live = json.loads(live_path.read_text())
    benches = ["toolbench", "tau_bench", "livecodebench", "swe_bench"]
    labels = ["ToolBench", r"$\tau$-bench", "LCB", "SWE"]
    allocators = ["hbac_d18", "type_prior", "uniform"]
    names = ["HBAC D18", "Type-prior", "Uniform"]
    colors = ["#2166ac", "#b2182b", "#878787"]

    x = np.arange(len(benches))
    width = 0.24
    fig, ax = plt.subplots(figsize=(3.6, 2.5))

    for i, (alloc, name, color) in enumerate(zip(allocators, names, colors)):
        vals = [
            live[alloc]["per_benchmark"][b]["pass_at_1"] * 100
            for b in benches
        ]
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=color, edgecolor="white", linewidth=0.4)

    ax.set_xticks(x, labels)
    ax.set_ylabel("pass@1 (%)")
    ax.set_title("Per-benchmark breakdown ($n{=}2000$, floor${=}400$)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(FIG / "per_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    floor_dose_response()
    per_benchmark()
    print(f"Wrote figures to {FIG}")
