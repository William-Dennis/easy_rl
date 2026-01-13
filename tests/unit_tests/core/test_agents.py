import numpy as np
from easy_marl.src.agents import PPOAgent

def get_agent(env):
    return PPOAgent(
        env=env,
        seed=123,
        learning_rate=1e-4,
        n_steps=32,
        batch_size=16,
        weight_decay=1e-5,
    )

class TestPPOAgentUnit:

    def test_initialization(self, mock_env):
        """PPOAgent initializes PPO model correctly, including weight decay."""
        agent = get_agent(mock_env)

        assert agent.env is mock_env
        assert agent.model is not None
        assert agent.model.n_steps == 32
        assert agent.model.batch_size == 16
        assert agent.model.seed == 123

        optimizer = agent.model.policy.optimizer
        for group in optimizer.param_groups:
            assert group["weight_decay"] == 1e-5

    def test_act_returns_valid_action(self, mock_env):
        """act() returns an action compatible with env action space."""
        agent = get_agent(mock_env)
        obs = mock_env.reset()

        action = agent.act(obs, deterministic=True)

        # flatten action if shape mismatch
        if hasattr(action, "shape") and len(action.shape) > 1:
            action = action.flatten()

        assert mock_env.action_space.contains(action)

    def test_act_is_deterministic(self, mock_env):
        """Deterministic act() returns same action for same obs."""
        agent = get_agent(mock_env)
        obs = mock_env.reset()

        a1 = agent.act(obs, deterministic=True)
        a2 = agent.act(obs, deterministic=True)

        assert np.allclose(a1, a2)

    def test_fixed_act_function(self, mock_env):
        """fixed_act_function returns a callable producing valid actions."""
        agent = get_agent(mock_env)
        obs = mock_env.reset()

        act_fn = agent.fixed_act_function(deterministic=True)
        action = act_fn(obs)

        assert callable(act_fn)

        if hasattr(action, "shape") and len(action.shape) > 1:
            action = action.flatten()

        assert mock_env.action_space.contains(action)

    def test_weight_decay_configuration(self, mock_env):
        """weight_decay is passed into optimizer kwargs."""
        agent = PPOAgent(mock_env, weight_decay=1e-5)

        optimizer = agent.model.policy.optimizer
        for group in optimizer.param_groups:
            assert group["weight_decay"] == 1e-5

    def test_train_runs_without_error(self, mock_env):
        """train() runs a minimal learning loop."""
        agent = PPOAgent(mock_env, n_steps=32, batch_size=16)

        # minimal steps to keep test fast
        agent.train(total_timesteps=64)

    def test_save_and_load(self, mock_env, tmp_path):
        """save() and load() correctly restore model."""
        agent = get_agent(mock_env)
        path = tmp_path / "ppo_agent"

        agent.save(str(path))

        new_agent = PPOAgent(mock_env)
        new_agent.load(str(path))

        obs = mock_env.reset()
        a1 = agent.act(obs, deterministic=True)
        a2 = new_agent.act(obs, deterministic=True)

        assert np.allclose(a1, a2)

    def test_save_to_bytes_and_load_from_bytes(self, mock_env):
        """save_to_bytes() and load_from_bytes() round-trip state."""
        agent = get_agent(mock_env)
        obs = mock_env.reset()
        action_before = agent.act(obs, deterministic=True)

        data = agent.save_to_bytes()
        assert isinstance(data, bytes)

        agent.load_from_bytes(data)
        action_after = agent.act(obs, deterministic=True)

        assert np.allclose(action_before, action_after)

    def test_from_bytes_classmethod(self, mock_env):
        """from_bytes() constructs a new agent with same behavior."""
        agent = get_agent(mock_env)
        obs = mock_env.reset()

        data = agent.save_to_bytes()
        new_agent = PPOAgent.from_bytes(data, mock_env)

        assert new_agent.env is mock_env
        assert new_agent.null_action is None

        a1 = agent.act(obs, deterministic=True)
        a2 = new_agent.act(obs, deterministic=True)

        assert np.allclose(a1, a2)
