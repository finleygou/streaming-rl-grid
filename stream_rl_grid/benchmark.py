"""Multi-seed comparison of TIDBD and fixed-step differential Sarsa."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .config import AppConfig, PROFILES
from .trainer import Trainer


def run_benchmark(profiles: List[str], seeds: List[int], steps: int, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    curves = {}
    for profile in profiles:
        for use_tidbd in (True, False):
            label = "TIDBD" if use_tidbd else "fixed-alpha"
            method_curves = []
            for seed in seeds:
                config = AppConfig()
                config.environment.profile = profile
                config.environment.seed = seed
                config.agent.use_tidbd = use_tidbd
                config.training.auto_checkpoint_steps = steps + 1
                trainer = Trainer(config, base_dir=output)
                snapshot = trainer.run_steps(steps)
                rows.append(
                    {
                        "profile": profile,
                        "method": label,
                        "seed": seed,
                        "steps": steps,
                        "average_reward": snapshot["average_reward"],
                        "reward_rate": snapshot["reward_rate"],
                        "goals_per_1000_steps": snapshot["goals_per_1000_steps"],
                        "collision_rate": snapshot["collision_rate"],
                        "abs_td_error": snapshot["abs_td_error"],
                    }
                )
                method_curves.append(snapshot["curves"])
            curves[(profile, label)] = method_curves

    csv_path = output / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(len(profiles), 1, figsize=(10, max(4, 3.5 * len(profiles))), squeeze=False)
    for row_index, profile in enumerate(profiles):
        axis = axes[row_index, 0]
        for label in ("TIDBD", "fixed-alpha"):
            entries = curves[(profile, label)]
            minimum = min(len(entry["steps"]) for entry in entries)
            x = np.asarray(entries[0]["steps"][:minimum])
            y = np.asarray([entry["average_reward"][:minimum] for entry in entries], dtype=float)
            mean = y.mean(axis=0)
            stderr = y.std(axis=0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.zeros_like(mean)
            axis.plot(x, mean, label=label)
            axis.fill_between(x, mean - 1.96 * stderr, mean + 1.96 * stderr, alpha=0.2)
        axis.set_title(profile)
        axis.set_xlabel("stream step")
        axis.set_ylabel("window average reward")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TIDBD against a fixed-step baseline")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output) if args.output else root / "benchmark_results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = run_benchmark(args.profiles, args.seeds, args.steps, output.resolve())
    print("Benchmark written to %s" % csv_path.parent)


if __name__ == "__main__":
    main()
