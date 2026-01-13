# Easy MARL

A generic, publication-ready Multi-Agent Reinforcement Learning framework. Decouples core logic from environments, supporting both continuous and discrete action spaces.

## Installation

```powershell
uv sync --extra dev
```

## Quick Start

Train agents using the high-level API. The framework supports both **Simultaneous Best Response (SBR)** and **Iterated Best Response (IBR)**.

```python
from easy_marl.core.training import parallel_train, sequential_train
from easy_marl.envs.electricity import ElectricityMarketEnv
from easy_marl.envs.snake import MultiSnakeEnv

# Example 1: Electricity Market (Continuous Action Space)
# Train 5 agents to bid in a day-ahead market
agents, stats = parallel_train(
    env_class=ElectricityMarketEnv,
    env_kwargs={"n_generators": 5},
    num_rounds=10,
    timesteps_per_agent=10_000,
    update_schedule="SBR"  # Simultaneous Best Response (Jacobi)
)

# Example 2: N-Player Snake / Tron (Discrete Action Space)
# Train 4 snakes to survive and box opponents in
agents, stats = sequential_train(
    env_class=MultiSnakeEnv,
    env_kwargs={"n_snakes": 4, "grid_size": 20},
    num_rounds=50,
    timesteps_per_agent=50_000,
    update_schedule="IBR"  # Iterated Best Response (Gauss-Seidel)
)
```

## Key Concepts

- **Core (`easy_marl.core`)**: Contains environment-agnostic `PPOAgent` and training loops.
- **Environments (`easy_marl.envs`)**:
  - `electricity`: Continuous bidding market.
  - `snake`: Discrete multi-agent Tron game.
- **Training Regimes**:
  - **SBR (Parallel)**: Agents train against frozen snapshots of opponents (stable in adversarial).
  - **IBR (Sequential)**: Agents train against live opponents (fast in potential games).

## Project Layout

```
easy_marl/
  src/
    core/               # Generic MARL Logic
    envs/
      electricity/      # Continuous Market Domain
      snake/            # Discrete Generic Domain
```
