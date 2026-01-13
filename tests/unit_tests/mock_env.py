"""
Mock environments for testing.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from easy_marl.src.core.base_env import BaseMARLEnv


class SimpleCoordinationEnv(BaseMARLEnv):
    """
    A simple coordination game for N agents.

    Dynamics:
    - Observations: Sampled from [0, 1].
    - Actions: Continuous [-1, 1].
    - Reward: Penalize variance (coordination) and magnitude.
    """

    def __init__(self, n_agents=2):
        self._n_agents = n_agents
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [self.observation_space.sample() for _ in range(self.n_agents)], {}

    def step(self, actions):
        actions = np.array(actions).flatten()
        if self.n_agents > 1:
            coordination_loss = np.var(actions)
        else:
            coordination_loss = 0.0
        magnitude_loss = np.mean(np.abs(actions))
        reward = -(coordination_loss + magnitude_loss)
        return (
            [self.observation_space.sample() for _ in range(self.n_agents)],
            [float(reward)] * self.n_agents,
            False,
            False,
            {},
        )

    def render(self):
        pass


class SingleAgentWrapper(gym.Env):
    """
    Wraps a BaseMARLEnv to look like a single-agent environment for one specific agent.
    """

    def __init__(self, env: BaseMARLEnv, agent_index: int):
        self.env = env
        self.agent_index = agent_index
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, seed=None, options=None):
        obs_list, info = self.env.reset(seed=seed, options=options)
        return obs_list[self.agent_index], info

    def step(self, action):
        all_actions = [self.action_space.sample() for _ in range(self.env.n_agents)]
        all_actions[self.agent_index] = action
        obs_list, rewards, term, trunc, info = self.env.step(all_actions)
        return obs_list[self.agent_index], rewards[self.agent_index], term, trunc, info
