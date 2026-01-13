"""
Integration tests for the generic training loop.
Verifies that sequential_train and parallel_train work with the shared mock environment.
"""

import pytest
import shutil
import os
from typing import List, Any
import numpy as np
import gymnasium as gym

from easy_marl.src.core.agents import PPOAgent, BaseAgent
from easy_marl.src.core.training import sequential_train, parallel_train
from tests.mock_env import SimpleCoordinationEnv, SingleAgentWrapper


def _make_factory_env(idx: int, agents_list: List[BaseAgent], seed: int) -> gym.Env:
    """
    Top-level factory function to ensure pickling works for multiprocessing.
    Wraps SimpleCoordinationEnv in SingleAgentWrapper for SB3 compatibility.
    """
    # Create the multi-agent env
    # Note: agents_list might be used in a real env for IBR/SBR logic.
    # SimpleCoordinationEnv ignores it currently, but accepts the arg.
    ma_env = SimpleCoordinationEnv(n_agents=2, agents=agents_list, seed=seed)
    
    # Wrap it to look like a specific single-agent environment for agent `idx`
    wrapper = SingleAgentWrapper(ma_env, agent_index=idx)
    return wrapper


@pytest.fixture
def clean_outputs():
    output_dir = "tests/test_outputs"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    yield output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def create_dummy_agents() -> List[PPOAgent]:
    """Helper to create initial agents."""
    agents = []
    # We need a dummy env for initialization of shapes
    # Just need observation_space and action_space
    dummy_env = SimpleCoordinationEnv(n_agents=2)
    # Wrap because PPO expects Single Agent view
    dummy_wrapper = SingleAgentWrapper(dummy_env, 0)
    
    for i in range(2):
        agent = PPOAgent(
            env=dummy_wrapper, 
            seed=i, 
            n_steps=64, 
            batch_size=64
        )
        agents.append(agent)
    return agents


def test_sequential_train_runs(clean_outputs):
    """Test that sequential_train runs without error."""
    agents = create_dummy_agents()
        
    trained_agents, info = sequential_train(
        agents=agents,
        env_factory=_make_factory_env,
        num_rounds=2,
        timesteps_per_agent=128,
        save_dir=clean_outputs,
        verbose=True
    )
    
    assert len(trained_agents) == 2
    assert info["num_rounds"] == 2
    
    # Verify outputs created
    assert os.path.exists(os.path.join(clean_outputs, "round_1/agent_0.zip"))
    assert os.path.exists(os.path.join(clean_outputs, "round_2/agent_1.zip"))


def test_parallel_train_runs(clean_outputs):
    """Test that parallel_train runs without error."""
    agents = create_dummy_agents()
        
    trained_agents, info = parallel_train(
        agents=agents,
        env_factory=_make_factory_env,
        num_rounds=2,
        timesteps_per_agent=128,
        save_dir=clean_outputs,
        verbose=True,
        n_workers=2
    )
    
    assert len(trained_agents) == 2
    assert info["parallel"] is True
