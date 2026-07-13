"""Streaming differential Sarsa(lambda) with per-feature TIDBD step sizes."""

from typing import Any, Dict, Sequence

import numpy as np

from .config import AgentConfig
from .tile_coder import DualTileCoder


class DifferentialSarsaTIDBD:
    """Linear continuing-control learner using replacing traces and TIDBD."""

    def __init__(self, coder: DualTileCoder, config: AgentConfig, seed: int = 0):
        config.validate()
        self.coder = coder
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(coder.size, dtype=np.float64)
        initial_alpha = config.effective_initial_step / coder.nominal_active_count
        self.beta = np.full(coder.size, np.log(initial_alpha), dtype=np.float64)
        self.h = np.zeros(coder.size, dtype=np.float64)
        self.trace = np.zeros(coder.size, dtype=np.float64)
        self.reward_rate = 0.0
        self.update_count = 0
        self.beta_clip_count = 0
        self.last_delta = 0.0

    @property
    def epsilon(self) -> float:
        return self.config.epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError("epsilon must lie in [0, 1].")
        self.config.epsilon = float(value)

    def value(self, observation: Sequence[int], action: int, readonly: bool = False) -> float:
        active = self.coder.active(observation, action, readonly=readonly)
        return float(self.weights[active].sum())

    def action_values(self, observation: Sequence[int], readonly: bool = False) -> np.ndarray:
        return np.asarray([self.value(observation, action, readonly=readonly) for action in range(5)], dtype=np.float64)

    def action_probabilities(self, observation: Sequence[int], readonly: bool = True) -> np.ndarray:
        """Return the exact epsilon-greedy policy without mutating the tile dictionary."""
        values = self.action_values(observation, readonly=readonly)
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))
        probabilities = np.full(5, self.config.epsilon / 5.0, dtype=np.float64)
        probabilities[best] += (1.0 - self.config.epsilon) / len(best)
        return probabilities

    def select_action(self, observation: Sequence[int]) -> int:
        if self.rng.random() < self.config.epsilon:
            return int(self.rng.integers(5))
        values = self.action_values(observation)
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=1e-12, atol=1e-12))
        return int(best[int(self.rng.integers(len(best)))])

    def update(
        self,
        observation: Sequence[int],
        action: int,
        reward: float,
        next_observation: Sequence[int],
        next_action: int,
    ) -> float:
        active = self.coder.active(observation, action)
        next_active = self.coder.active(next_observation, next_action)
        q = float(self.weights[active].sum())
        q_next = float(self.weights[next_active].sum())
        delta = float(reward - self.reward_rate + q_next - q)

        # Replacing eligibility traces. There is no gamma in the average-reward problem.
        self.trace *= self.config.lambda_
        self.trace[active] = 1.0

        if self.config.use_tidbd:
            proposed_beta = self.beta[active] + self.config.theta * delta * self.h[active]
            clipped_beta = np.clip(proposed_beta, self.config.beta_min, self.config.beta_max)
            self.beta_clip_count += int(np.count_nonzero(clipped_beta != proposed_beta))
            self.beta[active] = clipped_beta
            alpha = np.exp(self.beta)
            self.weights += alpha * delta * self.trace

            decay = np.ones_like(self.h)
            decay[active] = np.maximum(0.0, 1.0 - alpha[active] * self.trace[active])
            self.h = self.h * decay + alpha * delta * self.trace
        else:
            alpha = self.config.effective_initial_step / self.coder.nominal_active_count
            self.weights += alpha * delta * self.trace

        self.reward_rate += self.config.reward_rate_step * delta
        self.update_count += 1
        self.last_delta = delta
        self._check_finite()
        return delta

    def step_size_summary(self) -> Dict[str, float]:
        if self.config.use_tidbd:
            values = np.exp(self.beta)
        else:
            values = np.full(1, self.config.effective_initial_step / self.coder.nominal_active_count)
        return {
            "alpha_min": float(values.min()),
            "alpha_mean": float(values.mean()),
            "alpha_max": float(values.max()),
            "beta_clip_count": float(self.beta_clip_count),
        }

    def _check_finite(self) -> None:
        arrays = (self.weights, self.beta, self.h, self.trace)
        if not all(np.all(np.isfinite(array)) for array in arrays) or not np.isfinite(self.reward_rate):
            raise FloatingPointError("NaN or Inf detected in the learning state.")
        if np.max(np.abs(self.weights)) > 1e12 or abs(self.reward_rate) > 1e12:
            raise FloatingPointError("Learning state exceeded the configured numerical safety scale.")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.copy(),
            "beta": self.beta.copy(),
            "h": self.h.copy(),
            "trace": self.trace.copy(),
            "reward_rate": self.reward_rate,
            "update_count": self.update_count,
            "beta_clip_count": self.beta_clip_count,
            "last_delta": self.last_delta,
            "rng_state": self.rng.bit_generator.state,
            "coder": self.coder.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("weights", "beta", "h", "trace"):
            value = np.asarray(state[name], dtype=np.float64)
            if value.shape != (self.coder.size,):
                raise ValueError("Checkpoint %s has an incompatible shape." % name)
            setattr(self, name, value.copy())
        self.reward_rate = float(state["reward_rate"])
        self.update_count = int(state["update_count"])
        self.beta_clip_count = int(state["beta_clip_count"])
        self.last_delta = float(state["last_delta"])
        self.rng.bit_generator.state = state["rng_state"]
        self.coder.load_state_dict(state["coder"])
        self._check_finite()
