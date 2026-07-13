"""Tkinter control panel for configuring, training, saving, and inspecting the agent."""

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import AgentConfig, AppConfig, EnvironmentConfig, PROFILES, TrainingConfig, WIND_CHOICES
from .environment import ContinualWindyGridWorld
from .trainer import Trainer


Coord = Tuple[int, int]


class TrainingPanel:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Streaming Differential Sarsa + TIDBD - Continual Windy Grid")
        self.root.geometry("1450x880")
        self.base_dir = Path(__file__).resolve().parents[1]
        self.trainer: Optional[Trainer] = None
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.save_event = threading.Event()
        self.messages: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.preview_maps: Optional[List[Set[Coord]]] = None
        self.preview_context = 0
        self.selected_obstacle: Optional[Coord] = None
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self._canvas_geometry = (0.0, 0.0, 1.0)

        self.variables: Dict[str, tk.Variable] = {}
        self.metric_labels: Dict[str, ttk.Label] = {}
        self._build_layout()
        self._set_defaults(AppConfig())
        self.generate_preview()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_messages)

    def _build_layout(self) -> None:
        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        controls = ttk.Frame(outer, width=380)
        display = ttk.Frame(outer)
        outer.add(controls, weight=0)
        outer.add(display, weight=1)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True)
        env_tab, agent_tab, run_tab = ttk.Frame(notebook), ttk.Frame(notebook), ttk.Frame(notebook)
        notebook.add(env_tab, text="Environment")
        notebook.add(agent_tab, text="Agent")
        notebook.add(run_tab, text="Training")

        self._add_combo(env_tab, "Profile", "profile", PROFILES, 0)
        self._add_entry(env_tab, "Grid width", "width", 1)
        self._add_entry(env_tab, "Grid height", "height", 2)
        self._add_entry(env_tab, "Obstacle count", "obstacle_count", 3)
        self._add_entry(env_tab, "Obstacles (x,y; ...)", "obstacle_coordinates", 4)
        self._add_entry(env_tab, "Start (x,y)", "start_position", 5)
        self._add_entry(env_tab, "Goal (x,y)", "goal_position", 6)
        self._add_combo(env_tab, "Wind direction", "manual_wind_direction", WIND_CHOICES, 7)
        self._add_entry(env_tab, "Context maps", "num_contexts", 8)
        self._add_entry(env_tab, "Seed", "seed", 9)
        self._add_entry(env_tab, "Goal reward", "reward_goal", 10)
        self._add_entry(env_tab, "Collision reward", "reward_collision", 11)
        self._add_entry(env_tab, "Step reward", "reward_step", 12)
        self._add_entry(env_tab, "Max wind strength", "max_wind_strength", 13)
        self._add_entry(env_tab, "Wind/reward period", "wind_period", 14)
        self._add_entry(env_tab, "Goal move interval", "target_move_interval", 15)
        self._add_entry(env_tab, "Context switch interval", "context_switch_interval", 16)
        preview_row = ttk.Frame(env_tab)
        preview_row.grid(row=17, column=0, columnspan=2, sticky="ew", padx=6, pady=8)
        ttk.Button(preview_row, text="Generate maps", command=self.generate_preview).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(preview_row, text="Prev map", command=lambda: self._change_preview_context(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(preview_row, text="Next map", command=lambda: self._change_preview_context(1)).pack(side=tk.LEFT)

        self._add_entry(agent_tab, "Number of tilings / group", "num_tilings", 0)
        self._add_entry(agent_tab, "Tiles per dimension", "tiles_per_dimension", 1)
        self._add_entry(agent_tab, "IHT size", "iht_size", 2)
        self._add_entry(agent_tab, "Lambda", "lambda_", 3)
        self._add_entry(agent_tab, "Epsilon", "epsilon", 4)
        self._add_entry(agent_tab, "TIDBD theta", "theta", 5)
        self._add_entry(agent_tab, "Initial effective step", "effective_initial_step", 6)
        self._add_entry(agent_tab, "Reward-rate step", "reward_rate_step", 7)
        self._add_entry(agent_tab, "Beta minimum", "beta_min", 8)
        self._add_entry(agent_tab, "Beta maximum", "beta_max", 9)
        tidbd = tk.BooleanVar(value=True)
        self.variables["use_tidbd"] = tidbd
        ttk.Checkbutton(agent_tab, text="Use TIDBD (off = fixed-step baseline)", variable=tidbd).grid(
            row=10, column=0, columnspan=2, sticky="w", padx=6, pady=8
        )

        self._add_entry(run_tab, "Metric window", "metric_window", 0)
        self._add_entry(run_tab, "Chart points", "chart_points", 1)
        self._add_entry(run_tab, "UI update steps", "ui_update_steps", 2)
        self._add_entry(run_tab, "Auto-checkpoint steps", "auto_checkpoint_steps", 3)
        self._add_entry(run_tab, "Checkpoint folder", "checkpoint_dir", 4)
        self._add_entry(run_tab, "Log folder", "log_dir", 5)

        button_box = ttk.LabelFrame(controls, text="Controls")
        button_box.pack(fill=tk.X, pady=(8, 0))
        self.start_button = ttk.Button(button_box, text="Start new training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=4, pady=5, sticky="ew")
        self.pause_button = ttk.Button(button_box, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=4, pady=5, sticky="ew")
        self.save_button = ttk.Button(button_box, text="Save", command=self.request_save, state=tk.DISABLED)
        self.save_button.grid(row=1, column=0, padx=4, pady=5, sticky="ew")
        self.stop_button = ttk.Button(button_box, text="Stop (discard)", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=4, pady=5, sticky="ew")
        ttk.Button(button_box, text="Load checkpoint", command=self.load_training).grid(
            row=2, column=0, columnspan=2, padx=4, pady=5, sticky="ew"
        )
        self.apply_button = ttk.Button(button_box, text="Apply environment now", command=self.apply_live_environment)
        self.apply_button.grid(row=3, column=0, columnspan=2, padx=4, pady=5, sticky="ew")
        ttk.Button(button_box, text="Apply wind now", command=self.apply_live_wind).grid(
            row=4, column=0, columnspan=2, padx=4, pady=5, sticky="ew"
        )
        button_box.columnconfigure(0, weight=1)
        button_box.columnconfigure(1, weight=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var, wraplength=350).pack(fill=tk.X, pady=8)
        self.layout_var = tk.StringVar(value="Environment positions: -")
        ttk.Label(controls, textvariable=self.layout_var, wraplength=350).pack(fill=tk.X, pady=(0, 8))

        top = ttk.Frame(display)
        top.pack(fill=tk.BOTH, expand=True)
        self.grid_canvas = tk.Canvas(top, bg="white", highlightthickness=1, highlightbackground="#888")
        self.grid_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.grid_canvas.bind("<Button-1>", self._on_grid_click)

        metric_frame = ttk.LabelFrame(top, text="Live metrics", width=260)
        metric_frame.pack(side=tk.RIGHT, fill=tk.Y)
        metric_names = [
            ("step", "Step"), ("average_reward", "Window avg reward"),
            ("reward_rate", "Estimated reward rate"), ("goals_per_1000_steps", "Goals / 1000"),
            ("collision_rate", "Collision rate"), ("abs_td_error", "Mean |TD error|"),
            ("alpha_mean", "Mean TIDBD alpha"), ("alpha_max", "Max TIDBD alpha"),
            ("iht_used", "IHT used"), ("iht_collisions", "IHT collisions"),
            ("context_index", "Hidden context (log)"), ("wind_phase", "Wind phase (log)"),
            ("next_action", "Next action"),
        ]
        for row, (key, label) in enumerate(metric_names):
            ttk.Label(metric_frame, text=label + ":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            value_label = ttk.Label(metric_frame, text="-")
            value_label.grid(row=row, column=1, sticky="e", padx=5, pady=2)
            self.metric_labels[key] = value_label

        figure = Figure(figsize=(10, 2.8), dpi=100)
        self.reward_axis = figure.add_subplot(121)
        self.diagnostic_axis = figure.add_subplot(122)
        self.figure_canvas = FigureCanvasTkAgg(figure, master=display)
        self.figure_canvas.get_tk_widget().pack(fill=tk.X, pady=(6, 0))

    def _add_entry(self, parent: ttk.Frame, label: str, key: str, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="ew", padx=6, pady=4)
        parent.columnconfigure(1, weight=1)

    def _add_combo(self, parent: ttk.Frame, label: str, key: str, values, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18).grid(
            row=row, column=1, sticky="ew", padx=6, pady=4
        )

    def _set_defaults(self, config: AppConfig) -> None:
        values = {}
        values.update(config.environment.__dict__)
        values.update(config.agent.__dict__)
        values.update(config.training.__dict__)
        for key, variable in self.variables.items():
            if key in values:
                variable.set("" if values[key] is None else values[key])

    @staticmethod
    def _parse_coordinate(text: str, label: str) -> Optional[Coord]:
        value = text.strip()
        if not value:
            return None
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise ValueError("%s must use x,y format." % label)
        return int(parts[0]), int(parts[1])

    @classmethod
    def _parse_obstacles(cls, text: str) -> List[Coord]:
        if not text.strip():
            return []
        points = [cls._parse_coordinate(item, "Each obstacle") for item in text.split(";") if item.strip()]
        result = [point for point in points if point is not None]
        if len(set(result)) != len(result):
            raise ValueError("Obstacle coordinates must be unique.")
        return result

    @staticmethod
    def _format_coordinate(point: Coord) -> str:
        return "%d,%d" % point

    @classmethod
    def _format_obstacles(cls, points) -> str:
        return "; ".join(cls._format_coordinate(tuple(point)) for point in sorted(points))

    def _read_config(self) -> AppConfig:
        obstacles = self._parse_obstacles(self.variables["obstacle_coordinates"].get())
        obstacle_count = int(self.variables["obstacle_count"].get())
        if obstacles and len(obstacles) != obstacle_count:
            raise ValueError("Obstacle count is %d, but %d coordinates were entered." % (obstacle_count, len(obstacles)))
        profile = self.variables["profile"].get()
        num_contexts = int(self.variables["num_contexts"].get())
        expected_maps = num_contexts if profile in ("hidden_context", "combined") else 1
        env = EnvironmentConfig(
            width=int(self.variables["width"].get()), height=int(self.variables["height"].get()),
            obstacle_count=obstacle_count,
            num_contexts=num_contexts, profile=profile,
            seed=int(self.variables["seed"].get()), reward_goal=float(self.variables["reward_goal"].get()),
            reward_collision=float(self.variables["reward_collision"].get()),
            reward_step=float(self.variables["reward_step"].get()),
            max_wind_strength=int(self.variables["max_wind_strength"].get()),
            wind_period=int(self.variables["wind_period"].get()),
            target_move_interval=int(self.variables["target_move_interval"].get()),
            context_switch_interval=int(self.variables["context_switch_interval"].get()),
            start_position=list(self._parse_coordinate(self.variables["start_position"].get(), "Start"))
            if self.variables["start_position"].get().strip() else None,
            goal_position=list(self._parse_coordinate(self.variables["goal_position"].get(), "Goal"))
            if self.variables["goal_position"].get().strip() else None,
            manual_wind_direction=self.variables["manual_wind_direction"].get(),
        )
        preview_matches = (
            self.preview_maps is not None and len(self.preview_maps) == expected_maps
            and all(len(layout) == env.obstacle_count for layout in self.preview_maps)
            and set(obstacles) == self.preview_maps[self.preview_context % expected_maps]
        )
        if preview_matches:
            env.context_maps = [[list(point) for point in sorted(layout)] for layout in self.preview_maps]
        elif obstacles:
            env.context_maps = [[list(point) for point in obstacles] for _ in range(expected_maps)]
        agent = AgentConfig(
            num_tilings=int(self.variables["num_tilings"].get()),
            tiles_per_dimension=int(self.variables["tiles_per_dimension"].get()),
            iht_size=int(self.variables["iht_size"].get()), lambda_=float(self.variables["lambda_"].get()),
            epsilon=float(self.variables["epsilon"].get()), theta=float(self.variables["theta"].get()),
            effective_initial_step=float(self.variables["effective_initial_step"].get()),
            reward_rate_step=float(self.variables["reward_rate_step"].get()),
            beta_min=float(self.variables["beta_min"].get()), beta_max=float(self.variables["beta_max"].get()),
            use_tidbd=bool(self.variables["use_tidbd"].get()),
        )
        training = TrainingConfig(
            metric_window=int(self.variables["metric_window"].get()),
            chart_points=int(self.variables["chart_points"].get()),
            ui_update_steps=int(self.variables["ui_update_steps"].get()),
            auto_checkpoint_steps=int(self.variables["auto_checkpoint_steps"].get()),
            checkpoint_dir=self.variables["checkpoint_dir"].get(), log_dir=self.variables["log_dir"].get(),
        )
        config = AppConfig(env, agent, training)
        config.validate()
        return config

    def generate_preview(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            self.preview_maps = None
            self.variables["obstacle_coordinates"].set("")
            config = self._read_config()
            environment = ContinualWindyGridWorld(config.environment)
            self.preview_maps = [set(layout) for layout in environment.context_maps]
            self.preview_context = 0
            self.selected_obstacle = None
            self.variables["obstacle_count"].set(len(self.preview_maps[0]))
            self.variables["obstacle_coordinates"].set(self._format_obstacles(self.preview_maps[0]))
            self.variables["start_position"].set(self._format_coordinate(environment.start_position))
            self.variables["goal_position"].set(self._format_coordinate(environment.goal))
            snapshot = {
                "agent_state": environment.agent_state, "start_position": environment.start_position,
                "goal": environment.goal,
                "obstacles": sorted(self.preview_maps[0]), "dormant_obstacle": None,
                "wind": environment.wind_vector(environment.agent_state), "wind_phase": 0,
                "context_index": 0, "events": ["preview"],
            }
            self._draw_grid(snapshot, config.environment.width, config.environment.height)
            self.status_var.set("Maps generated. Click an obstacle, then a free cell, to relocate it.")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def _change_preview_context(self, amount: int) -> None:
        if not self.preview_maps:
            return
        self.preview_context = (self.preview_context + amount) % len(self.preview_maps)
        self.variables["obstacle_coordinates"].set(
            self._format_obstacles(self.preview_maps[self.preview_context])
        )
        config = self._read_config()
        snapshot = {
            "agent_state": (-1, -1), "start_position": self._parse_coordinate(
                self.variables["start_position"].get(), "Start"
            ) or (-1, -1),
            "goal": self._parse_coordinate(self.variables["goal_position"].get(), "Goal") or (-1, -1),
            "obstacles": sorted(self.preview_maps[self.preview_context]), "dormant_obstacle": None,
            "wind": (0, 0), "wind_phase": 0, "context_index": self.preview_context,
            "events": ["map preview"],
        }
        self.selected_obstacle = None
        self._draw_grid(snapshot, config.environment.width, config.environment.height)

    def _on_grid_click(self, event: tk.Event) -> None:
        if self.worker and self.worker.is_alive() and not self.pause_event.is_set() or not self.preview_maps:
            return
        ox, oy, cell = self._canvas_geometry
        if cell <= 0:
            return
        x, y = int((event.x - ox) // cell), int((event.y - oy) // cell)
        config = self._read_config()
        if not (0 <= x < config.environment.width and 0 <= y < config.environment.height):
            return
        point = (x, y)
        layout = self.preview_maps[self.preview_context]
        if self.selected_obstacle is None:
            if point in layout:
                self.selected_obstacle = point
                self.status_var.set("Obstacle selected; click a free destination cell.")
        else:
            if point not in layout:
                old = self.selected_obstacle
                layout.remove(old)
                layout.add(point)
                if not self._preview_maps_valid(config.environment):
                    layout.remove(point)
                    layout.add(old)
                    messagebox.showwarning("Disconnected map", "That move would disconnect the legal cells.")
                self.selected_obstacle = None
                self.variables["obstacle_coordinates"].set(self._format_obstacles(layout))
        snapshot = {
            "agent_state": tuple(self.last_snapshot.get("agent_state", (-1, -1))) if self.last_snapshot else (-1, -1),
            "start_position": self._parse_coordinate(self.variables["start_position"].get(), "Start") or (-1, -1),
            "goal": self._parse_coordinate(self.variables["goal_position"].get(), "Goal") or (-1, -1),
            "obstacles": sorted(layout),
            "dormant_obstacle": None, "wind": (0, 0), "wind_phase": 0,
            "context_index": self.preview_context, "events": ["map edited"],
        }
        self._draw_grid(snapshot, config.environment.width, config.environment.height)

    def apply_live_environment(self) -> None:
        try:
            config = self._read_config()
            obstacles = self._parse_obstacles(self.variables["obstacle_coordinates"].get())
            if len(obstacles) != config.environment.obstacle_count:
                raise ValueError("Enter exactly obstacle_count obstacle coordinates before applying.")
            start = self._parse_coordinate(self.variables["start_position"].get(), "Start")
            goal = self._parse_coordinate(self.variables["goal_position"].get(), "Goal")
            if start is None or goal is None:
                raise ValueError("Start and goal coordinates are required.")
            if self.trainer is None or not self.worker or not self.worker.is_alive():
                self.preview_maps = [set(obstacles) for _ in range(
                    config.environment.num_contexts if config.environment.profile in ("hidden_context", "combined") else 1
                )]
                environment = ContinualWindyGridWorld(config.environment)
                snapshot = {
                    "agent_state": environment.agent_state, "start_position": environment.start_position,
                    "goal": environment.goal, "obstacles": sorted(environment.active_obstacles),
                    "dormant_obstacle": None, "wind": environment.wind_vector(environment.agent_state),
                    "wind_phase": environment.wind_phase, "context_index": 0,
                    "manual_wind_direction": environment.config.manual_wind_direction,
                    "events": ["environment preview updated"],
                }
                self._draw_grid(snapshot, environment.width, environment.height)
                self.status_var.set("Environment preview updated; Start will use these values.")
                return
            if (config.environment.width, config.environment.height) != (
                self.trainer.config.environment.width, self.trainer.config.environment.height
            ):
                raise ValueError("Grid width/height cannot change during training; Stop and start a new run.")
            if not self.pause_event.is_set():
                raise ValueError("Pause training before applying map, start, goal, or profile changes.")
            snapshot = self.trainer.apply_environment_configuration(
                obstacles, start, goal, config.environment.manual_wind_direction, config.environment
            )
            self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
            self.preview_context = self.trainer.environment.context_index
            self._render_snapshot(snapshot)
            self.status_var.set("Environment changes applied at step %d." % self.trainer.step_count)
        except Exception as exc:
            messagebox.showerror("Cannot apply environment", str(exc))

    def apply_live_wind(self) -> None:
        try:
            direction = self.variables["manual_wind_direction"].get()
            strength = int(self.variables["max_wind_strength"].get())
            if strength < 0:
                raise ValueError("Max wind strength cannot be negative.")
            if self.trainer is None or not self.worker or not self.worker.is_alive():
                self.status_var.set("Wind selection will be used when training starts.")
                return
            snapshot = self.trainer.apply_wind(direction, strength)
            self._render_snapshot(snapshot)
            self.status_var.set("Wind changed immediately at step %d." % self.trainer.step_count)
        except Exception as exc:
            messagebox.showerror("Cannot apply wind", str(exc))

    def _preview_maps_valid(self, env_config: EnvironmentConfig) -> bool:
        try:
            copy_config = EnvironmentConfig(**env_config.__dict__)
            copy_config.context_maps = [[list(point) for point in sorted(layout)] for layout in self.preview_maps or []]
            ContinualWindyGridWorld(copy_config)
            return True
        except ValueError:
            return False

    def start_training(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            self.trainer = Trainer(self._read_config(), base_dir=self.base_dir)
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc))
            return
        self._launch_worker()

    def _launch_worker(self) -> None:
        self.stop_event.clear()
        self.pause_event.clear()
        self.save_event.clear()
        self.worker = threading.Thread(target=self._training_loop, name="stream-rl-training", daemon=True)
        self.worker.start()
        self.start_button.configure(state=tk.DISABLED)
        self.pause_button.configure(state=tk.NORMAL, text="Pause")
        self.save_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL)
        self.status_var.set("Training")
        if self.trainer is not None:
            self._render_snapshot(self.trainer.snapshot())

    def _training_loop(self) -> None:
        assert self.trainer is not None
        try:
            while not self.stop_event.is_set():
                if self.save_event.is_set():
                    path = self.trainer.save()
                    self.save_event.clear()
                    self.messages.put(("saved", path))
                if self.pause_event.is_set():
                    time.sleep(0.05)
                    continue
                snapshot = self.trainer.run_steps(
                    self.trainer.config.training.ui_update_steps, stop_event=self.stop_event
                )
                self.messages.put(("snapshot", snapshot))
            self.messages.put(("stopped", None))
        except Exception as exc:
            try:
                path = self.trainer.save()
            except Exception:
                path = None
            self.messages.put(("error", (exc, path)))

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Training")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            if self.trainer is not None:
                snapshot = self.trainer.snapshot()
                self._render_snapshot(snapshot)
                self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
                self.preview_context = self.trainer.environment.context_index
                current_layout = self.trainer.environment.context_maps[self.trainer.environment.context_index]
                self.variables["obstacle_count"].set(len(current_layout))
                self.variables["obstacle_coordinates"].set(self._format_obstacles(current_layout))
                self.variables["start_position"].set(self._format_coordinate(snapshot["start_position"]))
                self.variables["goal_position"].set(self._format_coordinate(snapshot["goal"]))
                self.variables["manual_wind_direction"].set(snapshot["manual_wind_direction"])
            self.status_var.set("Paused. Edit the environment, then click Apply environment now.")

    def request_save(self) -> None:
        self.save_event.set()
        self.status_var.set("Checkpoint requested...")

    def stop_training(self) -> None:
        self.stop_event.set()
        self.status_var.set("Stopping without saving after the current streaming update...")

    def load_training(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Training active", "Stop the current training before loading a checkpoint.")
            return
        path = filedialog.askopenfilename(
            title="Load exact-continuation checkpoint", initialdir=str(self.base_dir / "checkpoints"),
            filetypes=(("Pickle checkpoint", "*.pkl"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            self.trainer = Trainer.from_checkpoint(path, base_dir=self.base_dir)
            self._set_defaults(self.trainer.config)
            self.preview_maps = [set(layout) for layout in self.trainer.environment.context_maps]
            snapshot = self.trainer.snapshot()
            self._render_snapshot(snapshot)
            self.status_var.set("Checkpoint loaded at step %d. Starting exact continuation." % self.trainer.step_count)
            self._launch_worker()
        except Exception as exc:
            messagebox.showerror("Cannot load checkpoint", str(exc))

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "snapshot":
                    self._render_snapshot(payload)
                elif kind == "saved":
                    self.status_var.set("Saved: %s" % payload)
                elif kind == "stopped":
                    self.status_var.set("Stopped. Current training was not saved.")
                    self._set_idle_controls()
                elif kind == "error":
                    exc, path = payload
                    self._set_idle_controls()
                    messagebox.showerror("Training paused by safety check", "%s\nDiagnostic checkpoint: %s" % (exc, path))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _set_idle_controls(self) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.DISABLED)

    def _render_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.last_snapshot = snapshot
        assert self.trainer is not None
        self._draw_grid(snapshot, self.trainer.config.environment.width, self.trainer.config.environment.height)
        self.layout_var.set(
            "Start %s | Agent %s | Goal %s | Obstacles %s | Wind %s" % (
                tuple(snapshot.get("start_position", (-1, -1))), tuple(snapshot.get("agent_state", (-1, -1))),
                tuple(snapshot.get("goal", (-1, -1))), snapshot.get("obstacles", []),
                snapshot.get("manual_wind_direction", "auto"),
            )
        )
        for key, label in self.metric_labels.items():
            value = snapshot.get(key, "-")
            if isinstance(value, float):
                label.configure(text="%.6g" % value)
            else:
                label.configure(text=str(value))
        curves = snapshot["curves"]
        self.reward_axis.clear()
        self.reward_axis.plot(curves["steps"], curves["average_reward"], label="window reward")
        self.reward_axis.plot(curves["steps"], curves["reward_rate"], label="R-bar", alpha=0.8)
        self.reward_axis.set_title("Average reward")
        self.reward_axis.set_xlabel("stream step")
        self.reward_axis.grid(alpha=0.25)
        self.reward_axis.legend(fontsize=8)
        self.diagnostic_axis.clear()
        self.diagnostic_axis.plot(curves["steps"], curves["abs_td_error"], label="mean |delta|")
        self.diagnostic_axis.plot(curves["steps"], curves["alpha_mean"], label="mean alpha")
        self.diagnostic_axis.set_title("Adaptation diagnostics")
        self.diagnostic_axis.set_xlabel("stream step")
        self.diagnostic_axis.grid(alpha=0.25)
        self.diagnostic_axis.legend(fontsize=8)
        self.figure_canvas.draw_idle()

    def _draw_grid(self, snapshot: Dict[str, Any], width: int, height: int) -> None:
        canvas = self.grid_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        available_w = max(300, canvas.winfo_width())
        available_h = max(300, canvas.winfo_height())
        cell = max(8.0, min((available_w - 40) / width, (available_h - 60) / height))
        ox = (available_w - cell * width) / 2.0
        oy = (available_h - cell * height) / 2.0
        self._canvas_geometry = (ox, oy, cell)
        obstacles = {tuple(p) for p in snapshot.get("obstacles", [])}
        dormant = snapshot.get("dormant_obstacle")
        dormant = None if dormant is None else tuple(dormant)
        for y in range(height):
            for x in range(width):
                point = (x, y)
                fill = "#d9a441" if point in obstacles else "#f7f7f7"
                if point == dormant:
                    fill = "#f4dfad"
                if point == self.selected_obstacle and self.preview_context == snapshot.get("context_index", 0):
                    fill = "#d65ad1"
                x0, y0 = ox + x * cell, oy + y * cell
                canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=fill, outline="#9a9a9a")

        policies = snapshot.get("policy_probabilities")
        policy_color = "#86d7a1"
        if policies:
            directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
            for y in range(height):
                for x in range(width):
                    if (x, y) in obstacles or policies[y][x] is None:
                        continue
                    probabilities = policies[y][x]
                    cx, cy = ox + (x + 0.5) * cell, oy + (y + 0.5) * cell
                    for action, (dx, dy) in enumerate(directions):
                        length = 0.5 * cell * max(0.0, min(1.0, float(probabilities[action])))
                        if length > 0.5:
                            canvas.create_line(
                                cx, cy, cx + dx * length, cy + dy * length,
                                fill=policy_color, width=max(1, int(cell * 0.035)), arrow=tk.LAST,
                                arrowshape=(max(4, int(cell * 0.12)), max(5, int(cell * 0.14)),
                                            max(2, int(cell * 0.06))),
                            )
                    radius = 0.5 * cell * max(0.0, min(1.0, float(probabilities[4])))
                    if radius > 0.5:
                        canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                                           outline=policy_color, width=max(1, int(cell * 0.035)))

        start = tuple(snapshot.get("start_position", (-1, -1)))
        if 0 <= start[0] < width and 0 <= start[1] < height:
            x0, y0 = ox + start[0] * cell, oy + start[1] * cell
            canvas.create_rectangle(x0 + 0.08 * cell, y0 + 0.08 * cell, x0 + 0.92 * cell, y0 + 0.92 * cell,
                                    outline="#28a060", width=max(2, int(cell * 0.04)))
            canvas.create_text(x0 + 0.14 * cell, y0 + 0.12 * cell, text="S", anchor="nw", fill="#1b7947")
        goal = tuple(snapshot.get("goal", (-1, -1)))
        agent = tuple(snapshot.get("agent_state", (-1, -1)))
        if 0 <= goal[0] < width and 0 <= goal[1] < height:
            x0, y0 = ox + goal[0] * cell, oy + goal[1] * cell
            canvas.create_oval(x0 + 0.18 * cell, y0 + 0.18 * cell, x0 + 0.82 * cell, y0 + 0.82 * cell,
                               fill="#32b5d2", outline="")
        if 0 <= agent[0] < width and 0 <= agent[1] < height:
            x0, y0 = ox + agent[0] * cell, oy + agent[1] * cell
            canvas.create_text(x0 + cell / 2, y0 + cell / 2, text="A", fill="#2446d8",
                               font=("Segoe UI", max(10, int(cell * 0.42)), "bold"))
        wind = snapshot.get("wind", (0, 0))
        canvas.create_text(
            ox, max(12, oy - 22), anchor="w",
            text="map %s | wind %s | events: %s" % (
                snapshot.get("context_index", 0), wind, ", ".join(snapshot.get("events", [])) or "-"
            ), fill="#333",
        )

    def _draw_grid_legacy(self, snapshot: Dict[str, Any], width: int, height: int) -> None:
        canvas = self.grid_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        available_w = max(300, canvas.winfo_width())
        available_h = max(300, canvas.winfo_height())
        cell = max(8.0, min((available_w - 40) / width, (available_h - 60) / height))
        ox = (available_w - cell * width) / 2.0
        oy = (available_h - cell * height) / 2.0
        self._canvas_geometry = (ox, oy, cell)
        obstacles = {tuple(p) for p in snapshot.get("obstacles", [])}
        dormant = snapshot.get("dormant_obstacle")
        dormant = None if dormant is None else tuple(dormant)
        for y in range(height):
            for x in range(width):
                point = (x, y)
                fill = "#f7f7f7"
                if point in obstacles:
                    fill = "#d9a441"
                if point == dormant:
                    fill = "#f4dfad"
                if point == self.selected_obstacle and self.preview_context == snapshot.get("context_index", 0):
                    fill = "#d65ad1"
                x0, y0 = ox + x * cell, oy + y * cell
                canvas.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=fill, outline="#9a9a9a")
        goal = tuple(snapshot.get("goal", (-1, -1)))
        agent = tuple(snapshot.get("agent_state", (-1, -1)))
        if 0 <= goal[0] < width and 0 <= goal[1] < height:
            x0, y0 = ox + goal[0] * cell, oy + goal[1] * cell
            canvas.create_oval(x0 + 0.18 * cell, y0 + 0.18 * cell, x0 + 0.82 * cell, y0 + 0.82 * cell,
                               fill="#32b5d2", outline="")
        if 0 <= agent[0] < width and 0 <= agent[1] < height:
            x0, y0 = ox + agent[0] * cell, oy + agent[1] * cell
            canvas.create_text(x0 + cell / 2, y0 + cell / 2, text="★", fill="#2446d8",
                               font=("Segoe UI Symbol", max(10, int(cell * 0.45)), "bold"))
        wind = snapshot.get("wind", (0, 0))
        canvas.create_text(ox, max(12, oy - 22), anchor="w",
                           text="map %s | wind %s | events: %s" % (
                               snapshot.get("context_index", 0), wind, ", ".join(snapshot.get("events", [])) or "-"
                           ), fill="#333")

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Stop training", "Stop the current training without saving and close?"):
                return
            self.stop_event.set()
            self.worker.join(timeout=5.0)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrainingPanel(root)
    root.mainloop()


if __name__ == "__main__":
    main()
