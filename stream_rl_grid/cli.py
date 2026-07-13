"""Headless streaming trainer; Ctrl+C is a supported manual stopping mechanism."""

import argparse
from pathlib import Path

from .config import AppConfig, PROFILES
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming differential Sarsa(lambda) + TIDBD")
    parser.add_argument("--resume", type=str, help="Checkpoint to continue exactly")
    parser.add_argument("--profile", choices=PROFILES, default="combined")
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--fixed-alpha", action="store_true", help="Disable TIDBD baseline")
    parser.add_argument("--report-every", type=int, default=1_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    if args.resume:
        trainer = Trainer.from_checkpoint(args.resume, base_dir=base_dir)
        print("Loaded exact continuation at step %d" % trainer.step_count)
    else:
        config = AppConfig()
        config.environment.profile = args.profile
        config.environment.seed = args.seed
        config.environment.width = args.width
        config.environment.height = args.height
        config.environment.obstacle_count = args.obstacles
        config.agent.use_tidbd = not args.fixed_alpha
        trainer = Trainer(config, base_dir=base_dir)

    target = None if args.steps == 0 else trainer.step_count + args.steps
    try:
        while target is None or trainer.step_count < target:
            trainer.run_steps(min(args.report_every, (target - trainer.step_count) if target is not None else args.report_every))
            snapshot = trainer.snapshot()
            print(
                "step={step:.0f} avg_reward={average_reward:.4f} Rbar={reward_rate:.4f} "
                "goals/1k={goals_per_1000_steps:.2f} collision={collision_rate:.3f} alpha={alpha_mean:.3g}".format(
                    **snapshot
                )
            )
    except KeyboardInterrupt:
        print("\nManual stop requested.")
    finally:
        path = trainer.save()
        print("Exact-continuation checkpoint saved to %s" % path)


if __name__ == "__main__":
    main()
