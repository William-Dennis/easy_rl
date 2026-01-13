"""
Shared pytest fixtures for unit tests.
"""

import pytest
from easy_marl.examples.mock_env import SimpleCoordinationEnv, SingleAgentWrapper


@pytest.fixture
def mock_env():
    """Returns a fresh SimpleCoordinationEnv with 2 agents."""
    return SimpleCoordinationEnv(n_agents=2)


@pytest.fixture
def single_agent_env(mock_env):
    """Returns a Gym-compatible wrapper for Agent 0 of the mock env."""
    return SingleAgentWrapper(mock_env, agent_index=0)
