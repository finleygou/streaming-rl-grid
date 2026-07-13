"""Orchestration, CSV logging, and exact checkpoint restoration."""

import csv
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from .agent import DifferentialSarsaTIDBD
from .checkpoint import load_checkpoint, save_checkpoint
from .config import AppConfig
from .environment import ACTION_NAMES, ContinualWindyGridWorld
from .metrics import MetricsTracker
from .tile_coder import DualTileCoder


class Trainer:
    def __init__(self, config: AppConfig, base_dir: Optional[Union[str, Path]] = None, run_id: Optional[str] = None):
        config.validate()
        self.config = config
        self.base_dir = Path(base_dir or Path.cwd()).resolve()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.environment = ContinualWindyGridWorld(config.environment)
        self.coder = DualTileCoder(config.environment, config.agent)
        self.agent = DifferentialSarsaTIDBD(self.coder, config.agent, seed=config.environment.seed + 1)
        self.metrics = MetricsTracker(
            config.training.metric_window,
            config.training.chart_points,
            config.training.ui_update_steps,
        )
        self.current_observation = self.environment.observation()
        self.current_action = self.agent.select_action(self.current_observation)
        self.last_info: Dict[str, Any] = {}
        self.last_reward = 0.0
        self.last_checkpoint: Optional[Path] = None
        self._log_path = self.base_dir / config.training.log_dir / self.run_id / "metrics.csv"

    @property
    def step_count(self) -> int:
        return self.environment.step_count

    def step_once(self) -> Dict[str, Any]:
        next_observation, reward, terminated, truncated, info = self.environment.step(self.current_action)
        if terminated or truncated:
            raise RuntimeError("The continuing environment must never terminate or truncate.")
        next_action = self.agent.select_action(next_observation)
        delta = self.agent.update(
            self.current_observation,
            self.current_action,
            reward,
            next_observation,
            next_action,
        )
        self.current_observation = next_observation
        self.current_action = next_action
        self.last_info = info
        self.last_reward = reward
        alpha_summary = self.agent.step_size_summary()
        self.metrics.update(
            self.step_count,
            reward,
            delta,
            info,
            self.agent.reward_rate,
            alpha_summary["alpha_mean"],
        )
        if self.step_count % self.config.training.ui_update_steps == 0:
            self._append_log_row(delta, alpha_summary)
        if self.step_count % self.config.training.auto_checkpoint_steps == 0:
            self.save()
        return self.snapshot()

    def run_steps(self, count: int) -> Dict[str, Any]:
        snapshot = self.snapshot()
        for _ in range(int(count)):
            snapshot = self.step_once()
        return snapshot

    def snapshot(self) -> Dict[str, Any]:
        summary = self.metrics.summary(self.step_count)
        summary.update(self.agent.step_size_summary())
        summary.update(
            {
                "reward_rate": float(self.agent.reward_rate),
                "last_reward": float(self.last_reward),
                "last_delta": float(self.agent.last_delta),
                "agent_state": self.environment.agent_state,
                "goal": self.environment.goal,
                "obstacles": sorted(self.environment.active_obstacles),
                "dormant_obstacle": self.environment.dormant_obstacle,
                "context_index": self.environment.context_index,
                "wind_phase": self.environment.wind_phase,
                "reward_phase": self.environment.reward_phase,
                "wind": self.environment.wind_vector(self.environment.agent_state),
                "next_action": ACTION_NAMES[self.current_action],
                "events": list(self.environment.last_events),
                "curves": self.metrics.curves(),
                "iht_used": len(self.coder.iht.dictionary),
                "iht_size": self.coder.iht.size,
                "iht_collisions": self.coder.iht.overfull_count,
            }
        )
        return summary

    def default_checkpoint_path(self) -> Path:
        folder = self.base_dir / self.config.training.checkpoint_dir / self.run_id
        return folder / ("step-%012d.pkl" % self.step_count)

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        destination = Path(path).resolve() if path is not None else self.default_checkpoint_path()
        self.last_checkpoint = save_checkpoint(destination, self.state_dict())
        return self.last_checkpoint

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "compatibility": self._compatibility_signature(),
            "run_id": self.run_id,
            "environment": self.environment.state_dict(),
            "agent": self.agent.state_dict(),
            "metrics": self.metrics.state_dict(),
            "current_observation": tuple(self.current_observation),
            "current_action": int(self.current_action),
            "last_info": self.last_info,
            "last_reward": self.last_reward,
            "python_random_state": random.getstate(),
            "numpy_legacy_random_state": np.random.get_state(),
        }

    @classmethod
    def from_checkpoint(cls, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> "Trainer":
        state = load_checkpoint(path)
        config = AppConfig.from_dict(state["config"])
        trainer = cls(config, base_dir=base_dir, run_id=state["run_id"])
        if state.get("compatibility") != trainer._compatibility_signature():
            raise ValueError("Checkpoint feature or action configuration is incompatible.")
        trainer.environment.load_state_dict(state["environment"])
        trainer.agent.load_state_dict(state["agent"])
        trainer.metrics.load_state_dict(state["metrics"])
        trainer.current_observation = tuple(state["current_observation"])
        trainer.current_action = int(state["current_action"])
        trainer.last_info = dict(state["last_info"])
        trainer.last_reward = float(state["last_reward"])
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_legacy_random_state"])
        trainer.last_checkpoint = Path(path).resolve()
        return trainer

    def _compatibility_signature(self) -> Dict[str, Any]:
        return {
            "width": self.config.environment.width,
            "height": self.config.environment.height,
            "actions": list(ACTION_NAMES),
            "num_tilings": self.config.agent.num_tilings,
            "tiles_per_dimension": self.config.agent.tiles_per_dimension,
            "iht_size": self.config.agent.iht_size,
            "feature_groups": ["absolute_position", "relative_goal", "categorical_bias"],
        }

    def _append_log_row(self, delta: float, alpha: Dict[str, float]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._log_path.exists()
        with self._log_path.open("a", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "step", "reward", "average_reward", "reward_rate", "abs_td_error",
                "goals_per_1000_steps", "collision_rate", "alpha_min", "alpha_mean",
                "alpha_max", "delta", "context", "wind_phase", "goal_x", "goal_y", "events",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            summary = self.metrics.summary(self.step_count)
            writer.writerow(
                {
                    "step": self.step_count,
                    "reward": self.last_reward,
                    "average_reward": summary["average_reward"],
                    "reward_rate": self.agent.reward_rate,
                    "abs_td_error": summary["abs_td_error"],
                    "goals_per_1000_steps": summary["goals_per_1000_steps"],
                    "collision_rate": summary["collision_rate"],
                    "alpha_min": alpha["alpha_min"],
                    "alpha_mean": alpha["alpha_mean"],
                    "alpha_max": alpha["alpha_max"],
                    "delta": delta,
                    "context": self.environment.context_index,
                    "wind_phase": self.environment.wind_phase,
                    "goal_x": self.environment.goal[0],
                    "goal_y": self.environment.goal[1],
                    "events": "|".join(self.environment.last_events),
                }
            )
