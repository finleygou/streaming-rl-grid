"""Continuing-task metrics that do not rely on episode boundaries."""

from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np


class MetricsTracker:
    def __init__(self, window: int, chart_points: int, sample_interval: int):
        self.window = int(window)
        self.chart_points = int(chart_points)
        self.sample_interval = int(sample_interval)
        self.rewards: Deque[float] = deque(maxlen=self.window)
        self.abs_deltas: Deque[float] = deque(maxlen=self.window)
        self.collisions: Deque[int] = deque(maxlen=self.window)
        self.goals: Deque[int] = deque(maxlen=self.window)
        self.total_reward = 0.0
        self.total_goals = 0
        self.total_collisions = 0
        self.last_goal_step = 0
        self.goal_intervals: Deque[int] = deque(maxlen=self.window)
        self.curve_steps: Deque[int] = deque(maxlen=self.chart_points)
        self.curve_reward: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_reward_rate: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_abs_delta: Deque[float] = deque(maxlen=self.chart_points)
        self.curve_alpha_mean: Deque[float] = deque(maxlen=self.chart_points)

    def update(
        self,
        step: int,
        reward: float,
        delta: float,
        info: Dict[str, Any],
        reward_rate: float,
        alpha_mean: float,
    ) -> None:
        goal = int(bool(info.get("goal_reached", False)))
        collision = int(bool(info.get("collision", False)))
        self.rewards.append(float(reward))
        self.abs_deltas.append(abs(float(delta)))
        self.collisions.append(collision)
        self.goals.append(goal)
        self.total_reward += float(reward)
        self.total_goals += goal
        self.total_collisions += collision
        if goal:
            if self.last_goal_step > 0:
                self.goal_intervals.append(step - self.last_goal_step)
            self.last_goal_step = step
        if step % self.sample_interval == 0:
            self.curve_steps.append(step)
            self.curve_reward.append(self.window_average_reward)
            self.curve_reward_rate.append(float(reward_rate))
            self.curve_abs_delta.append(self.window_abs_delta)
            self.curve_alpha_mean.append(float(alpha_mean))

    @property
    def window_average_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def window_abs_delta(self) -> float:
        return float(np.mean(self.abs_deltas)) if self.abs_deltas else 0.0

    @property
    def window_goal_rate(self) -> float:
        return float(np.mean(self.goals)) if self.goals else 0.0

    @property
    def window_collision_rate(self) -> float:
        return float(np.mean(self.collisions)) if self.collisions else 0.0

    def summary(self, step: int) -> Dict[str, float]:
        return {
            "step": float(step),
            "average_reward": self.window_average_reward,
            "abs_td_error": self.window_abs_delta,
            "goals_per_1000_steps": self.window_goal_rate * 1000.0,
            "collision_rate": self.window_collision_rate,
            "mean_steps_between_goals": float(np.mean(self.goal_intervals)) if self.goal_intervals else 0.0,
            "total_goals": float(self.total_goals),
            "total_collisions": float(self.total_collisions),
        }

    def curves(self) -> Dict[str, List[float]]:
        return {
            "steps": list(self.curve_steps),
            "average_reward": list(self.curve_reward),
            "reward_rate": list(self.curve_reward_rate),
            "abs_td_error": list(self.curve_abs_delta),
            "alpha_mean": list(self.curve_alpha_mean),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "window": self.window,
            "chart_points": self.chart_points,
            "sample_interval": self.sample_interval,
            "rewards": list(self.rewards),
            "abs_deltas": list(self.abs_deltas),
            "collisions": list(self.collisions),
            "goals": list(self.goals),
            "total_reward": self.total_reward,
            "total_goals": self.total_goals,
            "total_collisions": self.total_collisions,
            "last_goal_step": self.last_goal_step,
            "goal_intervals": list(self.goal_intervals),
            "curves": self.curves(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if int(state["window"]) != self.window:
            raise ValueError("Checkpoint metric window is incompatible.")
        self.rewards = deque(state["rewards"], maxlen=self.window)
        self.abs_deltas = deque(state["abs_deltas"], maxlen=self.window)
        self.collisions = deque(state["collisions"], maxlen=self.window)
        self.goals = deque(state["goals"], maxlen=self.window)
        self.total_reward = float(state["total_reward"])
        self.total_goals = int(state["total_goals"])
        self.total_collisions = int(state["total_collisions"])
        self.last_goal_step = int(state["last_goal_step"])
        self.goal_intervals = deque(state["goal_intervals"], maxlen=self.window)
        curves = state["curves"]
        self.curve_steps = deque(curves["steps"], maxlen=self.chart_points)
        self.curve_reward = deque(curves["average_reward"], maxlen=self.chart_points)
        self.curve_reward_rate = deque(curves["reward_rate"], maxlen=self.chart_points)
        self.curve_abs_delta = deque(curves["abs_td_error"], maxlen=self.chart_points)
        self.curve_alpha_mean = deque(curves["alpha_mean"], maxlen=self.chart_points)
