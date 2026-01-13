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

    def __init__(self, n_agents=2, agents=None, seed=None):
        super().__init__()
        self._n_agents = n_agents
        self.agents = agents  # Store agents if provided
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
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