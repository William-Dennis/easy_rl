"""
Strict unit tests for PPOAgent.
Mocks external dependencies (stable_baselines3.PPO) to ensure fast, isolated execution.
"""

import pytest
from unittest.mock import MagicMock, patch
from easy_marl.src.core.agents import PPOAgent, BaseAgent
import gymnasium as gym


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_train_raises_not_implemented(self):
        """Verify BaseAgent.train() raises NotImplementedError."""

        class ConcreteAgent(BaseAgent):
            def act(self, obs, deterministic=True):
                return None

            def fixed_act_function(self, deterministic=True):
                return lambda x: None

        agent = ConcreteAgent()
        with pytest.raises(NotImplementedError, match="does not support training"):
            agent.train()


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
        seed = 123
        learning_rate = 1e-4

        _ = PPOAgent(env=mock_env, seed=seed, learning_rate=learning_rate, n_steps=1024)

        mock_ppo_class.assert_called_once()
        call_args = mock_ppo_class.call_args

        assert call_args[0][0] == "MlpPolicy"
        assert call_args[0][1] == mock_env
        assert call_args[1]["seed"] == seed
        assert call_args[1]["learning_rate"] == learning_rate
        assert call_args[1]["n_steps"] == 1024
        assert "policy_kwargs" in call_args[1]

    def test_act_calls_predict(self, mock_env, mock_ppo_class):
        """Test that act() delegates to model.predict()."""
        mock_model = mock_ppo_class.return_value
        expected_action = [0.5, -0.5]
        mock_model.predict.return_value = (expected_action, None)

        agent = PPOAgent(mock_env)
        obs = [1.0, 2.0]

        action = agent.act(obs, deterministic=True)

        mock_model.predict.assert_called_once_with(obs, deterministic=True)
        assert action == expected_action

    def test_weight_decay_configuration(self, mock_env, mock_ppo_class):
        """Test that weight_decay parameter constructs correct optimizer_kwargs."""
        _ = PPOAgent(mock_env, weight_decay=1e-5)

        call_args = mock_ppo_class.call_args
        policy_kwargs = call_args[1]["policy_kwargs"]

        assert "optimizer_kwargs" in policy_kwargs
        assert policy_kwargs["optimizer_kwargs"]["weight_decay"] == 1e-5

    def test_fixed_act_function_returns_callable(self, mock_env, mock_ppo_class):
        """Test that fixed_act_function returns a working callable."""
        mock_model = mock_ppo_class.return_value
        expected_action = [0.9]
        mock_model.predict.return_value = (expected_action, None)

        agent = PPOAgent(mock_env)

        act_fn = agent.fixed_act_function(deterministic=False)
        result = act_fn([1.0])

        assert callable(act_fn)
        mock_model.predict.assert_called_with([1.0], deterministic=False)
        assert result == expected_action

    def test_train_calls_learn(self, mock_env, mock_ppo_class):
        """Test that train() calls model.learn()."""
        mock_model = mock_ppo_class.return_value

        agent = PPOAgent(mock_env)
        agent.train(total_timesteps=1000)

        mock_model.learn.assert_called_once()
        call_args = mock_model.learn.call_args
        assert call_args[1]["total_timesteps"] == 1000

    @patch("stable_baselines3.common.callbacks.CheckpointCallback")
    def test_train_with_checkpoint(self, mock_cb_class, mock_env, mock_ppo_class):
        """Test that train() creates CheckpointCallback when path provided."""
        mock_model = mock_ppo_class.return_value

        agent = PPOAgent(mock_env)
        agent.train(
            total_timesteps=500, checkpoint_path="/tmp/ckpt", checkpoint_freq=100
        )

        mock_cb_class.assert_called_once_with(
            save_freq=100, save_path="/tmp/ckpt", name_prefix="ppo_agent"
        )
        mock_model.learn.assert_called_once()

    def test_save_calls_model_save(self, mock_env, mock_ppo_class):
        """Test that save() delegates to model.save()."""
        mock_model = mock_ppo_class.return_value

        agent = PPOAgent(mock_env)
        agent.save("/path/to/model")

        mock_model.save.assert_called_once_with("/path/to/model")

    def test_load_calls_model_load(self, mock_env, mock_ppo_class):
        """Test that load() uses PPO.load()."""
        agent = PPOAgent(mock_env)
        agent.load("/path/to/model")

        mock_ppo_class.load.assert_called_once_with("/path/to/model", env=mock_env)

    def test_save_to_bytes(self, mock_env, mock_ppo_class):
        """Test that save_to_bytes() serializes to buffer."""
        mock_model = mock_ppo_class.return_value

        agent = PPOAgent(mock_env)
        result = agent.save_to_bytes()

        mock_model.save.assert_called_once()
        assert isinstance(result, bytes)

    def test_load_from_bytes(self, mock_env, mock_ppo_class):
        """Test that load_from_bytes() deserializes from buffer."""
        agent = PPOAgent(mock_env)
        agent.load_from_bytes(b"fake_data")

        mock_ppo_class.load.assert_called()

    def test_from_bytes_class_method(self, mock_env, mock_ppo_class):
        """Test that from_bytes() creates new instance from serialized state."""
        agent = PPOAgent.from_bytes(b"fake_data", mock_env)

        assert agent.env == mock_env
        assert agent.null_action is None
        mock_ppo_class.load.assert_called()
