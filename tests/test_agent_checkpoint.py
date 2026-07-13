import tempfile
import unittest
from pathlib import Path

import numpy as np

from stream_rl_grid.config import AppConfig
from stream_rl_grid.trainer import Trainer


class AgentAndCheckpointTests(unittest.TestCase):
    def config(self):
        config = AppConfig()
        config.environment.width = 6
        config.environment.height = 5
        config.environment.obstacle_count = 2
        config.environment.profile = "combined"
        config.environment.wind_period = 11
        config.environment.target_move_interval = 7
        config.environment.context_switch_interval = 13
        config.agent.iht_size = 4096
        config.training.metric_window = 50
        config.training.ui_update_steps = 5
        config.training.auto_checkpoint_steps = 1_000_000
        return config

    def test_tile_coder_and_tidbd_remain_finite(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(200)
            self.assertTrue(np.all(np.isfinite(trainer.agent.weights)))
            self.assertTrue(np.all(np.isfinite(trainer.agent.beta)))
            self.assertEqual(trainer.step_count, 200)

    def test_checkpoint_continues_exactly_from_next_action(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(83)
            path = trainer.save(Path(folder) / "exact.pkl")
            expected = []
            for _ in range(31):
                trainer.step_once()
                expected.append((trainer.environment.observation(), trainer.current_action, trainer.last_reward))
            expected_weights = trainer.agent.weights.copy()
            expected_beta = trainer.agent.beta.copy()

            restored = Trainer.from_checkpoint(path, base_dir=folder)
            actual = []
            for _ in range(31):
                restored.step_once()
                actual.append((restored.environment.observation(), restored.current_action, restored.last_reward))
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(restored.agent.weights, expected_weights)
            np.testing.assert_array_equal(restored.agent.beta, expected_beta)
            self.assertEqual(restored.environment.state_dict()["rng_state"], trainer.environment.state_dict()["rng_state"])

    def test_policy_snapshot_is_normalized_without_allocating_visualization_features(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(10)
            used_before = len(trainer.coder.iht.dictionary)
            snapshot = trainer.snapshot()
            used_after = len(trainer.coder.iht.dictionary)
            self.assertEqual(used_after, used_before)
            for row in snapshot["policy_probabilities"]:
                for probabilities in row:
                    if probabilities is not None:
                        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_trainer_applies_live_environment_without_resetting_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(20)
            weights = trainer.agent.weights.copy()
            position = trainer.environment.agent_state
            previous_action = trainer.environment.previous_action
            current_action = trainer.current_action
            rng_state = trainer.agent.rng.bit_generator.state
            snapshot = trainer.apply_environment_configuration({(1, 1), (2, 1)}, (0, 0), (5, 4), "none")
            np.testing.assert_array_equal(trainer.agent.weights, weights)
            self.assertEqual(snapshot["agent_state"], position)
            self.assertEqual(trainer.environment.previous_action, previous_action)
            self.assertEqual(trainer.current_action, current_action)
            self.assertEqual(trainer.agent.rng.bit_generator.state, rng_state)
            self.assertEqual(snapshot["goal"], (5, 4))
            self.assertEqual(snapshot["manual_wind_direction"], "none")

    def test_live_wind_change_does_not_relocate_agent(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(20)
            position = trainer.environment.agent_state
            snapshot = trainer.apply_wind("left", 2)
            self.assertEqual(trainer.environment.agent_state, position)
            self.assertEqual(snapshot["manual_wind_direction"], "left")
            self.assertEqual(trainer.environment.config.max_wind_strength, 2)

    def test_environment_apply_preserves_policy_when_policy_state_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(self.config(), base_dir=folder)
            trainer.run_steps(40)
            before = trainer.snapshot()
            layout = set(trainer.environment.context_maps[trainer.environment.context_index])
            start = next(
                (x, y) for y in range(trainer.environment.height) for x in range(trainer.environment.width)
                if (x, y) not in layout and (x, y) != trainer.environment.goal
            )
            after = trainer.apply_environment_configuration(
                layout, start, trainer.environment.goal, "none"
            )
            self.assertEqual(after["policy_probabilities"], before["policy_probabilities"])


if __name__ == "__main__":
    unittest.main()
