"""
Base classes for Multi-Agent Reinforcement Learning (MARL) environments.
"""

from abc import abstractmethod
from typing import Tuple, Dict, Any, Optional
import gymnasium as gym


class BaseMARLEnv(gym.Env):
    """
    Abstract base class for Multi-Agent RL environments.

    This ensures a standard interface for:
    - n_agents: Property returning the number of agents.
    - step: Returning observations/rewards/dones for all agents.
    - reset: Returning initial observations for all agents.
    """

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Return the number of agents in the environment."""
        pass

    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            observations: Structure containing observations for all agents (usually List or Dict).
            info: Diagnostic information.
        """
        pass

    @abstractmethod
    def step(self, actions: Any) -> Tuple[Any, Any, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            actions: Actions for all agents (usually List or Dict).

        Returns:
            observations: New observations for all agents.
            rewards: Rewards for all agents.
            terminated: Whether the episode has naturally ended.
            truncated: Whether the episode was cut short (time limit).
            info: Diagnostic information.
        """
        pass
