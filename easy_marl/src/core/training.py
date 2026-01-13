"""
Generic MARL Training Logic.
Decoupled from specific environments and agent implementations where possible.
"""

import os
import random
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

from easy_marl.src.core.base_env import BaseMARLEnv
from easy_marl.src.core.agents import BaseAgent, PPOAgent

MAX_ENV_PROCS = psutil.cpu_count(logical=False) or os.cpu_count()
SAFE_MAX_ENV_PROCS = max(1, MAX_ENV_PROCS - 1)


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility across processes."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _train_agent_core(
    agent_index: int,
    round_idx: int,
    agents: List[
        PPOAgent
    ],  # Currently focused on PPOAgent, but could receive BaseAgent if they supported train()
    env_factory: Callable[[int, List[BaseAgent], int], BaseMARLEnv],
    timesteps_per_agent: int,
    save_dir: Optional[str],
    verbose: bool,
    N: int,
    seed: int,
) -> Dict:
    """
    Core training logic shared between sequential and parallel modes.

    Args:
        env_factory: Callable(agent_index, agents, seed) -> BaseMARLEnv
    """
    if verbose:
        mode_str = (
            "[Process]" if hasattr(os, "getpid") and os.getpid() != os.getppid() else ""
        )
        print(
            f"\n-> Training Agent {agent_index} ({timesteps_per_agent} timesteps) {mode_str}"
        )

    # Set all seeds immediately before training for reproducibility
    training_seed = seed + round_idx * N + agent_index
    set_all_seeds(training_seed)

    # Create environment for this agent using the factory
    env = env_factory(agent_index, agents, training_seed)

    # Assuming PPOAgent or compatible interface
    agents[agent_index].model.set_env(env)
    agents[agent_index].train(total_timesteps=timesteps_per_agent)

    # Save checkpoint
    if save_dir is not None:
        round_dir = os.path.join(save_dir, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        save_path = os.path.join(round_dir, f"agent_{agent_index}")
        agents[agent_index].save(save_path)
        if verbose:
            print(f"  [OK] Saved to {save_path}")

    # Generic evaluation could go here, but we'll return an empty dict for now
    # to keep it fully generic until the evaluation interface is defined in Phase 2 Plan.
    return {}


def sequential_train(
    agents: List[PPOAgent],
    env_factory: Callable[[int, List[BaseAgent], int], BaseMARLEnv],
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: Optional[str] = "outputs",
    verbose: bool = True,
    update_schedule: str = "SBR",
    update_probability: float = 1.0,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Single-process training with configurable update schedule (IBR or SBR).
    """
    # Validate update_schedule
    update_schedule = update_schedule.upper()
    if update_schedule not in ("IBR", "SBR"):
        raise ValueError(
            f"update_schedule must be 'IBR' or 'SBR', got '{update_schedule}'"
        )

    N = len(agents)

    # Training loop
    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": False,
        "update_schedule": update_schedule,
        "update_probability": update_probability,
    }

    # Ensure inertia decisions are reproducible
    random.seed(seed)

    all_results = []

    for round_idx in range(num_rounds):
        if verbose:
            print(f"\nTraining Round {round_idx + 1}/{num_rounds} [{update_schedule}]")

        # SBR: Snapshot all agent policies at the start of the round
        if update_schedule == "SBR":
            agents_snapshots = [agent.save_to_bytes() for agent in agents]

        trained_states = {}
        for i in range(N):
            # SBR: Restore all agents to round-start snapshot before each training
            if update_schedule == "SBR":
                for j, agent in enumerate(agents):
                    if j != i:  # Don't restore the agent we're about to train
                        agent.load_from_bytes(agents_snapshots[j])

            # Inertia check for SBR
            rng = random.Random(seed + round_idx * 100 + i)
            if (
                update_schedule == "SBR"
                and rng.random() > update_probability
                and round_idx > 0
            ):
                if verbose:
                    print(f"  [Inertia] Skipping update for Agent {i}")
                # Keep the old state (snapshot)
                trained_states[i] = agents_snapshots[i]
                continue

            eval_metrics = _train_agent_core(
                agent_index=i,
                round_idx=round_idx,
                agents=agents,
                env_factory=env_factory,
                timesteps_per_agent=timesteps_per_agent,
                save_dir=save_dir,
                verbose=verbose,
                N=N,
                seed=seed,
            )

            # SBR: Save the trained state to apply at round end
            if update_schedule == "SBR":
                trained_states[i] = agents[i].save_to_bytes()

            all_results.append(
                {
                    "round": round_idx + 1,
                    "agent": i,
                    "metrics": eval_metrics,
                }
            )

        # SBR: Synchronize all trained policies at round end
        if update_schedule == "SBR":
            for i in range(N):
                agents[i].load_from_bytes(trained_states[i])

    return agents, training_info


def train_single_agent_worker(
    agent_index: int,
    round_idx: int,
    agent_state: bytes,
    agents_states: List[bytes],
    env_factory: Callable[
        [int, List[BaseAgent], int], BaseMARLEnv
    ],  # BEWARE: pickled callable must be importable or simple
    timesteps_per_agent: int,
    save_dir: Optional[str],
    verbose: bool,
    N: int,
    seed: int,
) -> Tuple[int, bytes, Dict]:
    """Worker function for parallel training."""

    # Reconstruct agents
    # IMPORTANT: We need to recreate the environment context for these agents.
    # But from_bytes requires an env. We can create a dummy env or use the factory.
    # To avoid pickling issues and complexity, we reconstruct agents with a factory-produced env.

    # Temporary reconstruction with factory.
    # NOTE: The factory must handle creating envs with UNINITIALIZED agents if possible,
    # or we handle this bootstrap carefully.
    # PPOAgent.from_bytes needs an env.

    # Bootstrap: create dummy agents first? No, we have states.
    # We will assume env_factory can handle a partial list or we construct shells.

    # Strategy:
    # 1. Create env for agent i using factory (passing empty list initially to safe init?)
    #    Actually the factory expects `agents`. Ideally the factory constructs the env which HOLDS the agents.
    #    For PPOAgent.from_bytes(state, env), we need the env.

    # Let's try to pass a lightweight env or use the factory for each agent.
    # But the env needs THE OTHER agents to step.

    # Rehydrating ALL agents:
    # First pass: create uninitialized shells if needed, or just create them sequentially?
    # We need to construct them so we can pass them to the env factory.

    # Limitation: This approach requires PPOAgent to be importable.

    # 1. We create the environment for the TARGET agent (agent_index).
    # But wait, that environment needs REFERENCES to the other agents to define their behavior (fixed_act_function).
    # Catch-22: Agents need Env, Env needs Agents.
    # Resolution: PPOAgent(env) sets self.env. Env receives agents list.
    # We can init agents with env=None, then set_env later?
    # SB3 PPO requires env at init? Yes usually.

    # Correct PPOAgent.from_bytes implementation:
    # instance = cls.__new__(cls)
    # instance.env = env
    # instance.model = PPO.load(..., env=env)

    # So we MUST have the env ready.
    # And the Generic Env Helper likely needs "other agents policies".

    # In parallel_train/worker:
    # We need to reconstruct all agents.
    # We can create a dummy env or use the real one.
    # If using real one:
    #   env = env_factory(i, agents_list???, seed)
    #   agent = PPOAgent.from_bytes(state, env)

    # To break the cycle:
    # 1. Create a "Lazy" env or init agents with DummyEnv.
    # 2. Or, if `env_factory` creates the env, it probably needs the agents to query their actions.

    # We will assume we can:
    # 1. Create all agents using a dummy env or None (if SB3 supports loading without env then set_env).
    # 2. Create the real 'training env' for agent_index using the list of agents.
    # 3. Set the real env on the target agent.

    # Let's try loading with env=None for peers.

    # Rehydrate peers
    rehydrated_agents = []
    for i, state in enumerate(agents_states):
        # Peers don't need a functional step() env, just one to satisfy SB3 if they predict.
        # But if they predict, they are just policies.
        # We'll try loading with None first.
        agent = PPOAgent.from_bytes(state, env=None)
        rehydrated_agents.append(agent)

    # Now create the training environment for our target learner
    # The factory takes (agent_index, agents_list, seed)
    training_env = env_factory(agent_index, rehydrated_agents, seed)

    # Now attach this env to the target agent
    target_agent = rehydrated_agents[agent_index]
    target_agent.model.set_env(training_env)
    target_agent.env = training_env  # update ref

    # Train
    eval_metrics = _train_agent_core(
        agent_index=agent_index,
        round_idx=round_idx,
        agents=rehydrated_agents,
        env_factory=lambda *args: training_env,  # Already created
        timesteps_per_agent=timesteps_per_agent,
        save_dir=save_dir,
        verbose=verbose,
        N=N,
        seed=seed,
    )

    trained_state = target_agent.save_to_bytes()
    return agent_index, trained_state, eval_metrics


def parallel_train(
    agents: List[PPOAgent],
    env_factory: Callable[[int, List[BaseAgent], int], BaseMARLEnv],
    num_rounds: int = 3,
    timesteps_per_agent: int = 5000,
    seed: int = 42,
    save_dir: Optional[str] = "outputs",
    verbose: bool = True,
    n_workers: int = SAFE_MAX_ENV_PROCS,
    update_probability: float = 1.0,
) -> Tuple[List[PPOAgent], Dict]:
    """
    Simultaneous Best Response (SBR) training with parallel execution.
    """
    N = len(agents)
    n_workers = min(n_workers, N)

    training_info = {
        "N": N,
        "num_rounds": num_rounds,
        "timesteps_per_agent": timesteps_per_agent,
        "total_timesteps": N * num_rounds * timesteps_per_agent,
        "seed": seed,
        "parallel": True,
        "n_workers": n_workers,
        "update_probability": update_probability,
    }

    random.seed(seed)
    all_results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for round_idx in range(num_rounds):
            if verbose:
                print(f"\nTraining Round {round_idx + 1}/{num_rounds} [PARALLEL SBR]")

            # Serialize all agent states
            agents_states = [agent.save_to_bytes() for agent in agents]
            futures = []

            for i in range(N):
                # Inertia check consistent with sequential (SBR)
                rng = random.Random(seed + round_idx * 100 + i)
                if rng.random() > update_probability and round_idx > 0:
                    if verbose:
                        print(f"  [Inertia] Skipping update for Agent {i}")
                    continue

                future = executor.submit(
                    train_single_agent_worker,
                    agent_index=i,
                    round_idx=round_idx,
                    agent_state=agents_states[i],
                    agents_states=agents_states,
                    env_factory=env_factory,
                    timesteps_per_agent=timesteps_per_agent,
                    save_dir=save_dir,
                    verbose=verbose,
                    N=N,
                    seed=seed,
                )
                futures.append(future)

            # Collect results
            for future in as_completed(futures):
                idx, trained_state, metrics = future.result()
                # Update agent with new state
                agents[idx].load_from_bytes(trained_state)
                # Store metrics
                all_results.append(
                    {"round": round_idx + 1, "agent": idx, "metrics": metrics}
                )

    return agents, training_info
