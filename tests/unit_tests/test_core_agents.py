"""
Unit tests for the generic core agents.
Verifies that agents can be initialized, act, and saved/loaded without electricity-specific dependencies.
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from easy_marl.src.core.agents import PPOAgent
from easy_marl.src.core.base_env import BaseMARLEnv


class SimpleMockEnv(BaseMARLEnv):
    """
    A simple multi-agent environment for testing purposes.
    Observation: Box(1) - just a random number.
    Action: Box(1) - just a number.
    """

    def __init__(self, n_agents=2):
        self._n_agents = n_agents
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def reset(self, seed=None, options=None):
        # Returns a LIST of observations
        return [self.observation_space.sample() for _ in range(self.n_agents)], {}

    def step(self, actions):
        # Returns LISTS of obs, rewards
        obs = [self.observation_space.sample() for _ in range(self.n_agents)]
        rewards = [0.0] * self.n_agents
        # Simple step: never terminates naturally
        return obs, rewards, False, False, {}

    def render(self):
        pass


class SingleAgentWrapper(gym.Env):
    """
    Wraps a BaseMARLEnv to look like a single-agent environment for one specific agent.
    This is necessary because PPOAgent uses stable-baselines3, which expects standard Gym API
    (scalar/dict observation, not list of observations).
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
        # In a real scenario, we'd need to supply actions for OTHER agents too.
        # For this mock test, the environment ignores input actions, so we pass a dummy list.
        # We just need to make sure 'actions' passed to step includes our agent's action.

        # Create dummy actions for everyone
        all_actions = [self.action_space.sample() for _ in range(self.env.n_agents)]
        all_actions[self.agent_index] = action

        obs_list, rewards, term, trunc, info = self.env.step(all_actions)

        return (
            obs_list[self.agent_index],
            rewards[self.agent_index],
            term,
            trunc,
            info,
        )


@pytest.fixture
def mock_env():
    return SimpleMockEnv(n_agents=2)


@pytest.fixture
def single_agent_env(mock_env):
    """Returns a Gym-compatible wrapper for Agent 0."""
    return SingleAgentWrapper(mock_env, agent_index=0)


class TestPPOAgent:
    def test_initialization(self, single_agent_env):
        agent = PPOAgent(single_agent_env)
        assert agent.model is not None

    def test_act(self, single_agent_env):
        agent = PPOAgent(single_agent_env)
        obs = single_agent_env.observation_space.sample()
        action = agent.act(obs)
        assert action.shape == single_agent_env.action_space.shape

    def test_training(self, single_agent_env):
        agent = PPOAgent(single_agent_env)
        # Train for a minimal amount of steps to ensure no crash
        agent.train(total_timesteps=100)

    def test_serialization(self, single_agent_env, mock_env):
        # Note: We need to pass the same wrapper architecture to from_bytes/load
        agent = PPOAgent(single_agent_env, seed=42)
        agent.train(total_timesteps=100)

        # Serialize
        data = agent.save_to_bytes()

        # Deserialize into new agent.
        # We need a fresh wrapper for the new agent to avoid sharing state if any existed (mock env is stateless but good practice)
        new_wrapper = SingleAgentWrapper(mock_env, agent_index=0)
        new_agent = PPOAgent.from_bytes(data, new_wrapper)

        # Check parameter equality
        params1 = dict(agent.model.policy.named_parameters())
        params2 = dict(new_agent.model.policy.named_parameters())

        for name, param in params1.items():
            assert torch.equal(param, params2[name]), f"Parameter {name} mismatch"

    def test_fixed_act_function(self, single_agent_env):
        agent = PPOAgent(single_agent_env)
        act_fn = agent.fixed_act_function()

        obs = single_agent_env.observation_space.sample()
        action = act_fn(obs)
        assert action.shape == single_agent_env.action_space.shape
