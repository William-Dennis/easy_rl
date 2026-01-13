"""
Unit tests for the generic core agents.
Verifies that agents can be initialized, act, and saved/loaded using the shared mock environment.
"""

import torch
from easy_marl.src.core.agents import PPOAgent
from tests.unit_tests.mock_env import SingleAgentWrapper

# Note: mock_env and single_agent_env fixtures are automatically available from conftest.py


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
        # We need a fresh wrapper for the new agent to avoid sharing state if any existed
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
