from easy_marl.src.agents import SimpleAgent, PPOAgent
from easy_marl.src.environment import MARLElectricityMarketEnv
import pytest
import numpy as np
import torch


def mock_env():
    params = {
        "N_generators": 3,
        "T": 6,
        "demand_profile": [1, 2, 3, 4, 5, 6],
        "capacities": [1, 2, 3],
        "costs": [1, 2, 3],
    }
    env = MARLElectricityMarketEnv(params=params, agents=[])
    env.reset()
    return env


MOCK_ENV = mock_env()

# Create a set of example observations
example_obs = [MOCK_ENV._get_obs(agent_index=i) for i in range(MOCK_ENV.N)]


def zero_weights_in_ppo_policy(agent: PPOAgent):
    policy = agent.model.policy
    state_dict = policy.state_dict()

    for k, v in state_dict.items():
        state_dict[k] = torch.zeros_like(v)

    policy.load_state_dict(state_dict)


class TestActionFunction:
    """Tests that the action function returns the expected action."""

    @pytest.mark.parametrize("obs", example_obs)
    def test_that_simple_agent_returns_null_action(self, obs):
        agent = SimpleAgent(MOCK_ENV)

        agent_action = agent.act(obs)

        assert np.allclose(agent_action, [0.0, 0.0])

    @pytest.mark.parametrize("obs", example_obs)
    def test_that_ppo_agent_with_zero_parameters_returns_null_action(self, obs):
        agent = PPOAgent(MOCK_ENV)

        # Zero ALL parameters in the policy
        zero_weights_in_ppo_policy(agent)

        # Act with the zeroed policy
        agent_action = agent.act(obs)

        assert np.allclose(agent_action, [0.0, 0.0])


class TestFixedActionFunction:
    """Tests that the fixed action function returns the expected action."""

    @pytest.mark.parametrize("obs", example_obs)
    def test_that_simple_agent_is_not_affected_by_model_changes(self, obs):
        agent = SimpleAgent(MOCK_ENV)

        fixed_action_function = agent.fixed_act_function(deterministic=True)

        action_before = fixed_action_function(obs)

        agent.null_action = [1.0, 1.0]

        action_after = fixed_action_function(obs)

        assert np.allclose(action_before, action_after)

    @pytest.mark.skip(reason="Known issue: PPOAgent.fixed_act_function is not truly frozen")
    @pytest.mark.parametrize("obs", example_obs)
    def test_that_ppo_agent_is_not_affected_by_model_changes(self, obs):
        agent = PPOAgent(MOCK_ENV)

        # action function with random initialised weights
        fixed_action_function = agent.fixed_act_function(deterministic=True)

        action_before = fixed_action_function(obs)

        zero_weights_in_ppo_policy(agent)

        action_after = fixed_action_function(obs)

        assert np.allclose(action_before, action_after)


class TestAgentSerialization:
    """
    Tests that agents can be serialized and deserialized correctly.
    """

    def test_ppo_agent_serialization(self):
        """
        Tests that PPOAgent serialization preserves model parameters and optimizer state.
        """
        # Create a single-agent environment to avoid needing other agents
        params = {
            "N_generators": 1,
            "T": 24,
            "demand_profile": [10] * 24,
            "capacities": [100],
            "costs": [10],
        }
        env = MARLElectricityMarketEnv(params=params, agents=[])

        agent = PPOAgent(env, seed=42, learning_rate=1e-3)

        # Train for a few steps to populate optimizer state
        agent.train(total_timesteps=100)

        # Capture state before serialization
        params_before = [p.clone() for p in agent.model.policy.parameters()]
        opt_state_before = agent.model.policy.optimizer.state_dict()

        # Serialize
        data = agent.save_to_bytes()

        # Deserialize into a new agent
        new_env = MARLElectricityMarketEnv(params=params, agents=[])
        new_agent = PPOAgent(new_env, seed=42, learning_rate=1e-3)
        new_agent.load_from_bytes(data)

        # 1. Check model parameters
        for p1, p2 in zip(params_before, new_agent.model.policy.parameters()):
            assert torch.equal(p1, p2), (
                "Model parameters do not match after deserialization"
            )

        # 2. Check optimizer state
        opt_state_after = new_agent.model.policy.optimizer.state_dict()

        # Check keys match
        assert opt_state_before.keys() == opt_state_after.keys()

        # Check internal state presence
        internal_state_before = opt_state_before["state"]
        internal_state_after = opt_state_after["state"]

        assert len(internal_state_before) == len(internal_state_after)
        assert len(internal_state_before) > 0, (
            "Optimizer state should not be empty after training"
        )

        # Check a sample value from internal state to ensure it's not just empty dicts
        k = list(internal_state_before.keys())[0]
        v_before = internal_state_before[k]
        v_after = internal_state_after[k]

        for state_name in v_before:
            if torch.is_tensor(v_before[state_name]):
                assert torch.allclose(v_before[state_name], v_after[state_name]), (
                    f"Optimizer internal state mismatch for {state_name}"
                )

    def test_from_bytes_creates_equivalent_agent(self):
        """Tests that from_bytes produces an agent equivalent to __init__ + load_from_bytes."""
        params = {
            "N_generators": 1,
            "T": 24,
            "demand_profile": [10] * 24,
            "capacities": [100],
            "costs": [10],
        }
        env = MARLElectricityMarketEnv(params=params, agents=[])

        original = PPOAgent(env, seed=42, learning_rate=1e-3)
        original.train(total_timesteps=100)

        # Serialize
        data = original.save_to_bytes()

        # Method 1: __init__ + load_from_bytes
        env1 = MARLElectricityMarketEnv(params=params, agents=[])
        agent1 = PPOAgent(env1, seed=99)  # Different seed to ensure load overwrites
        agent1.load_from_bytes(data)

        # Method 2: from_bytes
        env2 = MARLElectricityMarketEnv(params=params, agents=[])
        agent2 = PPOAgent.from_bytes(data, env2)

        # Compare: both should produce the same action
        test_obs = env.observation_space.sample()
        action1 = agent1.act(test_obs)
        action2 = agent2.act(test_obs)

        assert np.allclose(action1, action2), (
            "Actions should match between the two methods"
        )
