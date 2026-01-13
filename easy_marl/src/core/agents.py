"""
Generic Agent implementations for Multi-Agent Reinforcement Learning.
"""

import torch
from abc import ABC, abstractmethod
from stable_baselines3 import PPO
from typing import Callable, Any, Optional, List, Tuple
import gymnasium as gym


class BaseAgent(ABC):
    """Abstract base class for all MARL agents."""

    def __init__(self, env: Optional[gym.Env] = None):
        self.env = env
        # Default null action. Environments should handle specific null action semantics.
        # For continuous spaces, this might be a zero vector.
        self.null_action: Any = None

    @abstractmethod
    def act(self, obs: Any, deterministic: bool = True) -> Any:
        """Return an action given an observation."""
        pass

    @abstractmethod
    def fixed_act_function(self, deterministic: bool = True) -> Callable:
        """Return a frozen action function for use by other agents."""
        pass

    def train(self, *args, **kwargs):
        """Optional: Override if the agent can learn."""
        raise NotImplementedError("This agent type does not support training.")


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent.
    Uses a neural network policy to learn optimal strategies.
    Wraps stable_baselines3.PPO.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 42,
        learning_rate: float = 3e-4,
        gamma: float = 0.98,
        clip_range: float = 0.2,
        n_steps: int = 2048,
        batch_size: int = 64,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Initialize PPO agent.

        Args:
            env: Gym environment instance
            seed: Random seed for reproducibility
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            clip_range: PPO clipping parameter
            n_steps: Number of steps to collect before update
            batch_size: Minibatch size for updates
            hidden_sizes: Tuple of hidden layer sizes
            weight_decay: L2 regularization coefficient
        """
        super().__init__(env)

        # Neural network architecture
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes)),
        )

        if weight_decay > 0:
            # The default optimizer is Adam, we are just adding weight_decay
            policy_kwargs["optimizer_kwargs"] = {"weight_decay": weight_decay}

        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=learning_rate,
            gamma=gamma,
            clip_range=clip_range,
            n_steps=n_steps,
            batch_size=batch_size,
            seed=seed,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )

    def train(
        self,
        total_timesteps: int = 50000,
        checkpoint_path: Optional[str] = None,
        checkpoint_freq: int = 1000,
        callbacks: Optional[List] = None,
    ):
        """Train the agent using PPO."""
        from stable_baselines3.common.callbacks import CheckpointCallback

        callback_list = callbacks if callbacks is not None else []

        if checkpoint_path is not None:
            checkpoint_cb = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_path,
                name_prefix="ppo_agent",
            )
            callback_list.append(checkpoint_cb)

        self.model.learn(total_timesteps=total_timesteps, callback=callback_list)

    def act(self, obs: Any, deterministic: bool = True) -> Any:
        """Return action for given observation."""
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def fixed_act_function(self, deterministic: bool = True) -> Callable:
        """
        Return a frozen action function.
        Note: SB3 models don't easily support deep-copying just the prediction function
        without overhead, so this binds to the current model instance.
        """

        def act_fn(obs):
            action, _ = self.model.predict(obs, deterministic=deterministic)
            return action

        return act_fn

    def save(self, path: str):
        """Save model to disk."""
        self.model.save(path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = PPO.load(path, env=self.env)

    def save_to_bytes(self) -> bytes:
        """Serialize agent state to bytes (includes optimizer state)."""
        import io

        buffer = io.BytesIO()
        self.model.save(buffer)
        return buffer.getvalue()

    def load_from_bytes(self, data: bytes):
        """Load serialized state into this existing instance."""
        import io

        buffer = io.BytesIO(data)
        self.model = PPO.load(buffer, env=self.env)

    @classmethod
    def from_bytes(cls, data: bytes, env: gym.Env) -> "PPOAgent":
        """Create a new agent instance directly from serialized state."""
        import io

        # Create uninitialized instance
        instance = cls.__new__(cls)
        instance.env = env
        instance.null_action = None

        buffer = io.BytesIO(data)
        instance.model = PPO.load(buffer, env=env)

        return instance
