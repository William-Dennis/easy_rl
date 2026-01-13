# Easy MARL Refactoring Plan: Decoupled Framework

This document outlines the detailed plan to refactor the `easy_marl` package. The goal is to separate the core Multi-Agent Reinforcement Learning (MARL) logic from the specific Electricity Market application, creating a truly reusable "publication ready" framework.

## Critique of Previous Architecture

The prior codebase had significant structural issues preventing it from being a generic framework:

- **Circular Dependencies**: Core `src` components imported from `examples`, which is an architectural violation.
- **Tight Coupling**: Training loops mixed generic SBR/IBR logic with domain-specific evaluation metrics (electricity prices).
- **Missing Abstractions**: Lack of a strong `BaseMARLEnv` contract.
- **Test Coverage**: Tests relied on the electricity environment, preventing independent verification of the core.

## Proposed Directory Structure

```
easy_marl/
├── src/
│   ├── core/               # Generic MARL components (Environment Agnostic)
│   │   ├── __init__.py
│   │   ├── agents.py       # BaseAgent, PPOAgent (Clean dependencies)
│   │   ├── base_env.py     # BaseMARLEnv (Abstract Base Class)
│   │   └── training.py     # sequential_train, parallel_train (Generic)
│   ├── envs/               # Domain-specific Environments
│   │   ├── __init__.py
│   │   └── electricity/    # Electricity Market Domain
│   │       ├── __init__.py
│   │       ├── market_env.py   # MARLElectricityMarketEnv
│   │       ├── market_logic.py # Performance critical logic (was in examples)
│   │       └── observators.py  # specific feature engineering
│   │   └── snake/          # [NEW] Discrete Tron/Snake Game
│   │       ├── __init__.py
│   │       └── snake_env.py
│   └── utils/
│       └── serialization.py # Helper for agent saving/loading
├── examples/
│   └── bidding/            # Usage example
│       ├── main.py         # Entry point
│       └── configs.py      # Specific parameter generation
└── tests/                  # Updated tests
```

## Migration Phases

### Phase 1: Core Foundation & Generic Tests

Focus: Build the `core` module and verify it with generic tests before moving complex domain logic.

1. **`easy_marl/src/core/base_env.py`**
    - Define `BaseMARLEnv(gym.Env)` abstract base class.
    - Enforce contracts for `n_agents`, `reset()`, and `step()`.

2. **`easy_marl/src/core/agents.py`**
    - Move `BaseAgent` and `PPOAgent` here.
    - Remove all electricity-specific references.

3. **`tests/unit_tests/test_core_training.py`**
    - Implement a `SimpleMockEnv` (e.g., a simple coordination game).
    - Verify `sequential_train` and `parallel_train` work with this mock environment.
    - This validates that the core training loop is truly generic.

### Phase 2: Refactoring Training Logic

Focus: Decouple the training loops from domain-specific metrics.

1. **Generic Training Loop (`src/core/training.py`)**
    - Port `sequential_train` and `parallel_train` logic.
    - Remove `make_competitive_params` dependency (use generic env factories).
    - Generalized evaluation: Replace hardcoded `market_prices` tracking with generic reward/metric callbacks.

### Phase 3: Migrating Electricity Market

Focus: Re-integrate the electricity market as a sub-package `easy_marl.envs.electricity`.

1. **`easy_marl/src/envs/electricity/`**
    - Move `market_logic.py` (numba logic) here from examples.
    - Move `MARLElectricityMarketEnv` here, inheriting from `BaseMARLEnv`.
    - Move `observators.py` here.

2. **Update Examples**
    - Refactor `examples/bidding/` to import from the new `core` and `envs.electricity` locations.

## Verification Plan

### Automated Tests

1. **Core Verification**: `pytest tests/unit_tests/test_core_training.py`
    - Must pass using only the generic mock environment.
2. **Regression Verification**: `pytest tests/unit_tests/test_agents.py`
    - Verify PPO agents still function correctly after move.
3. **End-to-End**: Run `python examples/bidding/main.py`
    - Verify the electricity market example runs without error using the new structure.

### Manual Verification

- **Code Review**: Ensure `src/core` contains ZERO imports from `src/envs` or `examples`.
