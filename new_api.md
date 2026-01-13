# New API Structure: Decoupled MARL Framework

This document outlines the proposed organization of the `easy_marl` package after decoupling the core reinforcement learning logic from the specific electricity market application.

## Core Framework (`easy_marl/src/core`)

The core contains generic, environment-agnostic MARL components.

- **`agents.py`**: Contains `BaseAgent` and `PPOAgent`. These classes handle policy inference and training logic without assuming anything about the environment's domain.
- **`training.py`**: Implements the multi-round training loops (`sequential_train`, `parallel_train`, `auto_train`). It operates on any Gymnasium-compatible environment.
- **`base_env.py`**: Provides `BaseMARLEnv`, an abstract base class for multi-agent environments, standardizing how agents interact with the simulation.

## Environment Library (`easy_marl/src/envs`)

Specific simulations are grouped into sub-packages.

### Electricity Market (`easy_marl/src/envs/electricity`)

- **`market_env.py`**: The `MARLElectricityMarketEnv` implementation, focusing on the bidding simulation logic.
- **`market_logic.py`**: Performance-critical market clearing functions (using Numba).
- **`observators.py`**: Domain-specific observation builders (feature engineering) for electricity market data.

## Example Workflows (`easy_marl/examples`)

- **`bidding/`**: A high-level example showing how to initialize the electricity environment and run the core training routines.

## Benefits

- **Extensibility**: Easily add new environments (e.g., traffic control, supply chain) by implementing a new sub-package in `envs/`.
- **Maintainability**: Core RL logic is isolated from simulation-specific physics/rules.
- **Testability**: The core can be tested using simple mock environments.
