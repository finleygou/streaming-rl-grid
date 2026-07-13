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


if __name__ == "__main__":
    unittest.main()
