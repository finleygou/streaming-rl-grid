"""Serializable configuration objects for the continual windy-grid experiment."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


PROFILES = ("stationary", "seasonal_wind", "moving_goal", "hidden_context", "combined", "customize")
WIND_CHOICES = ("auto", "up", "right", "down", "left", "none")


@dataclass
class EnvironmentConfig:
    width: int = 5
    height: int = 5
    obstacle_count: int = 3
    num_contexts: int = 3
    profile: str = "customize"
    seed: int = 0
    reward_goal: float = 10.0
    reward_collision: float = -5.0
    reward_step: float = -1.0
    max_wind_strength: int = 1
    wind_period: int = 2_000
    target_move_interval: int = 500
    context_switch_interval: int = 3_000
    context_maps: Optional[List[List[List[int]]]] = None
    goal_path: Optional[List[List[int]]] = None
    start_position: Optional[List[int]] = None
    goal_position: Optional[List[int]] = None
    manual_wind_direction: str = "none"

    def validate(self) -> None:
        if self.width < 3 or self.height < 3:
            raise ValueError("Grid width and height must both be at least 3.")
        if self.profile not in PROFILES:
            raise ValueError("Unknown non-stationarity profile: %s" % self.profile)
        if self.obstacle_count < 0 or self.obstacle_count > self.width * self.height - 2:
            raise ValueError("Obstacle count leaves fewer than two legal cells.")
        if self.num_contexts < 1:
            raise ValueError("num_contexts must be positive.")
        if self.max_wind_strength < 0:
            raise ValueError("max_wind_strength cannot be negative.")
        if self.manual_wind_direction not in WIND_CHOICES:
            raise ValueError("Unknown wind direction: %s" % self.manual_wind_direction)
        for name in ("start_position", "goal_position"):
            point = getattr(self, name)
            if point is not None:
                if len(point) != 2 or not (0 <= int(point[0]) < self.width and 0 <= int(point[1]) < self.height):
                    raise ValueError("%s must be an in-bounds [x, y] coordinate." % name)
        if self.start_position is not None and self.goal_position is not None:
            if tuple(self.start_position) == tuple(self.goal_position):
                raise ValueError("Start and goal coordinates must differ.")
        for name in ("wind_period", "target_move_interval", "context_switch_interval"):
            if getattr(self, name) <= 0:
                raise ValueError("%s must be positive." % name)


@dataclass
class AgentConfig:
    num_tilings: int = 8
    tiles_per_dimension: int = 8
    iht_size: int = 65_536
    lambda_: float = 0.8
    epsilon: float = 0.1
    theta: float = 0.01
    effective_initial_step: float = 0.1
    reward_rate_step: float = 0.01
    beta_min: float = -20.0
    beta_max: float = 0.0
    use_tidbd: bool = True

    def validate(self) -> None:
        if self.num_tilings < 1:
            raise ValueError("num_tilings must be positive.")
        if self.tiles_per_dimension < 2:
            raise ValueError("tiles_per_dimension must be at least 2.")
        if self.iht_size < 128:
            raise ValueError("iht_size must be at least 128.")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError("lambda must lie in [0, 1].")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1].")
        if self.theta < 0.0 or self.effective_initial_step <= 0.0:
            raise ValueError("TIDBD step-size parameters must be positive.")
        if self.reward_rate_step <= 0.0:
            raise ValueError("reward_rate_step must be positive.")
        if self.beta_min >= self.beta_max:
            raise ValueError("beta_min must be smaller than beta_max.")


@dataclass
class TrainingConfig:
    metric_window: int = 1_000
    chart_points: int = 1_500
    ui_update_steps: int = 50
    auto_checkpoint_steps: int = 10_000
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    def validate(self) -> None:
        for name in ("metric_window", "chart_points", "ui_update_steps", "auto_checkpoint_steps"):
            if getattr(self, name) <= 0:
                raise ValueError("%s must be positive." % name)


@dataclass
class AppConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        self.environment.validate()
        self.agent.validate()
        self.training.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        result = cls(
            environment=EnvironmentConfig(**data.get("environment", {})),
            agent=AgentConfig(**data.get("agent", {})),
            training=TrainingConfig(**data.get("training", {})),
        )
        result.validate()
        return result
