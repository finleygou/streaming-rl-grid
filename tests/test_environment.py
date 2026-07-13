import unittest

from stream_rl_grid.config import EnvironmentConfig
from stream_rl_grid.environment import ContinualWindyGridWorld


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **changes):
        config = EnvironmentConfig(
            width=5,
            height=5,
            obstacle_count=0,
            profile="stationary",
            max_wind_strength=0,
            context_maps=[[]],
            seed=3,
        )
        for key, value in changes.items():
            setattr(config, key, value)
        return ContinualWindyGridWorld(config)

    def test_goal_is_rewarded_and_teleported_without_termination(self):
        env = self.make_env()
        env.goal = (2, 2)
        env.agent_state = (1, 2)
        observation, reward, terminated, truncated, info = env.step(1)
        self.assertEqual(reward, env.config.reward_goal)
        self.assertTrue(info["goal_reached"])
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotEqual(env.agent_state, env.goal)
        self.assertEqual(observation[:4], env.observation()[:4])

    def test_invalid_action_stays_and_receives_collision_reward(self):
        env = self.make_env()
        env.agent_state = (0, 0)
        before = env.agent_state
        _, reward, terminated, _, info = env.step(3)
        self.assertEqual(env.agent_state, before)
        self.assertEqual(reward, env.config.reward_collision)
        self.assertTrue(info["collision"])
        self.assertFalse(terminated)

    def test_stay_action_is_still_affected_by_wind(self):
        env = self.make_env(max_wind_strength=1)
        env.agent_state = (2, 3)
        env.goal = (4, 4)
        env.step(4)
        self.assertEqual(env.agent_state, (2, 2))

    def test_new_context_obstacle_is_dormant_until_agent_leaves(self):
        env = self.make_env(
            obstacle_count=1,
            profile="hidden_context",
            num_contexts=2,
            context_switch_interval=2,
            context_maps=[[[4, 0]], [[2, 2]]],
        )
        env.agent_state = (2, 2)
        env.goal = (4, 4)
        env.step(4)
        env.step(4)
        self.assertEqual(env.dormant_obstacle, (2, 2))
        self.assertNotIn((2, 2), env.active_obstacles)
        env.step(1)
        self.assertIsNone(env.dormant_obstacle)


if __name__ == "__main__":
    unittest.main()
