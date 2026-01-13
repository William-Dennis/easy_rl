"""
Strict unit tests for PPOAgent.
Mocks external dependencies (stable_baselines3.PPO) to ensure fast, isolated execution.
"""

import pytest
from unittest.mock import MagicMock, patch
from easy_marl.src.core.agents import PPOAgent
import gymnasium as gym


class TestPPOAgentUnit:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock(spec=gym.Env)
        return env

    @pytest.fixture
    def mock_ppo_class(self):
        with patch("easy_marl.src.core.agents.PPO") as mock:
            yield mock

    def test_initialization_calls_ppo_correctly(self, mock_env, mock_ppo_class):
        """Test that PPO is initialized with correct parameters."""
        # Arrange
        seed = 123
        learning_rate = 1e-4

        # Act
        _ = PPOAgent(env=mock_env, seed=seed, learning_rate=learning_rate, n_steps=1024)

        # Assert
        mock_ppo_class.assert_called_once()
        call_args = mock_ppo_class.call_args

        # Check positional args
        assert call_args[0][0] == "MlpPolicy"
        assert call_args[0][1] == mock_env

        # Check kwargs
        assert call_args[1]["seed"] == seed
        assert call_args[1]["learning_rate"] == learning_rate
        assert call_args[1]["n_steps"] == 1024
        assert "policy_kwargs" in call_args[1]

    def test_act_calls_predict(self, mock_env, mock_ppo_class):
        """Test that act() delegates to model.predict()."""
        # Arrange
        mock_model = mock_ppo_class.return_value
        # Configure predict return value: (action, state)
        expected_action = [0.5, -0.5]
        mock_model.predict.return_value = (expected_action, None)

        agent = PPOAgent(mock_env)
        obs = [1.0, 2.0]

        # Act
        action = agent.act(obs, deterministic=True)

        # Assert
        mock_model.predict.assert_called_once_with(obs, deterministic=True)
        assert action == expected_action

    def test_weight_decay_configuration(self, mock_env, mock_ppo_class):
        """Test that weight_decay parameter constructs correct optimizer_kwargs."""
        # Act
        _ = PPOAgent(mock_env, weight_decay=1e-5)

        # Assert
        call_args = mock_ppo_class.call_args
        policy_kwargs = call_args[1]["policy_kwargs"]

        assert "optimizer_kwargs" in policy_kwargs
        assert policy_kwargs["optimizer_kwargs"]["weight_decay"] == 1e-5

    def test_fixed_act_function_returns_callable(self, mock_env, mock_ppo_class):
        """Test that fixed_act_function returns a working callable."""
        # Arrange
        mock_model = mock_ppo_class.return_value
        expected_action = [0.9]
        mock_model.predict.return_value = (expected_action, None)

        agent = PPOAgent(mock_env)

        # Act
        act_fn = agent.fixed_act_function(deterministic=False)
        result = act_fn([1.0])

        # Assert
        assert callable(act_fn)
        mock_model.predict.assert_called_with([1.0], deterministic=False)
        assert result == expected_action
