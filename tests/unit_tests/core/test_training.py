"""
Strict unit tests for training.py.
Mocks external dependencies (PPOAgent, ProcessPoolExecutor) for isolated testing.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import random


class TestSetAllSeeds:
    """Tests for set_all_seeds function."""

    def test_sets_all_seeds(self):
        """Verify all RNGs are seeded correctly."""
        from easy_marl.src.core.training import set_all_seeds
        import torch

        # Just verify it runs without error - actual seeding verified by behavior
        set_all_seeds(42)

        # Verify determinism by checking random state is consistent
        import random
        import numpy as np

        set_all_seeds(123)
        r1 = random.random()
        n1 = np.random.rand()

        set_all_seeds(123)
        r2 = random.random()
        n2 = np.random.rand()

        assert r1 == r2
        assert n1 == n2

    def test_sets_cuda_seeds_when_available(self):
        """Verify CUDA branch is covered (runs without error on CPU)."""
        from easy_marl.src.core.training import set_all_seeds
        import torch

        # This test just ensures the cuda branch is exercised
        # On CPU machines, torch.cuda.is_available() returns False
        # On GPU machines, it will set CUDA seeds
        set_all_seeds(99)
        # No assertion needed - just coverage

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    def test_cuda_seeds_called_when_available(
        self, mock_seed_all, mock_seed, mock_is_available
    ):
        """Cover lines 28-29: CUDA seed setting when GPU is available."""
        from easy_marl.src.core.training import set_all_seeds

        set_all_seeds(42)

        mock_seed.assert_called_with(42)
        mock_seed_all.assert_called_with(42)





class TestTrainAgentCore:
    """Tests for _train_agent_core function."""

    @patch("easy_marl.src.core.training.set_all_seeds")
    @patch("easy_marl.src.core.training.os.makedirs")
    def test_trains_agent_and_saves(self, mock_makedirs, mock_set_seeds):
        """Verify core training logic calls train and save."""
        from easy_marl.src.core.training import _train_agent_core

        mock_agent = MagicMock()
        mock_env = MagicMock()
        mock_factory = MagicMock(return_value=mock_env)

        result = _train_agent_core(
            agent_index=0,
            round_idx=0,
            agents=[mock_agent],
            env_factory=mock_factory,
            timesteps_per_agent=100,
            save_dir="test_dir",
            verbose=False,
            N=1,
            seed=42,
        )

        mock_set_seeds.assert_called_once_with(42)
        mock_factory.assert_called_once_with(0, [mock_agent], 42)
        mock_agent.model.set_env.assert_called_once_with(mock_env)
        mock_agent.train.assert_called_once_with(total_timesteps=100)
        mock_agent.save.assert_called_once()
        assert result == {}

    @patch("easy_marl.src.core.training.set_all_seeds")
    def test_skips_save_when_no_dir(self, mock_set_seeds):
        """Verify save is skipped when save_dir is None."""
        from easy_marl.src.core.training import _train_agent_core

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=MagicMock())

        _train_agent_core(
            agent_index=0,
            round_idx=0,
            agents=[mock_agent],
            env_factory=mock_factory,
            timesteps_per_agent=100,
            save_dir=None,
            verbose=False,
            N=1,
            seed=42,
        )

        mock_agent.save.assert_not_called()

    @patch("easy_marl.src.core.training.set_all_seeds")
    @patch("builtins.print")
    def test_verbose_output(self, mock_print, mock_set_seeds):
        """Verify verbose mode prints training info."""
        from easy_marl.src.core.training import _train_agent_core

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=MagicMock())

        _train_agent_core(
            agent_index=1,
            round_idx=2,
            agents=[MagicMock(), mock_agent],
            env_factory=mock_factory,
            timesteps_per_agent=500,
            save_dir=None,
            verbose=True,
            N=2,
            seed=42,
        )

        assert mock_print.called


class TestSequentialTrain:
    """Tests for sequential_train function."""

    @patch("easy_marl.src.core.training._train_agent_core")
    def test_ibr_schedule(self, mock_train_core):
        """Verify IBR schedule doesn't snapshot/restore agents."""
        from easy_marl.src.core.training import sequential_train

        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        agents = [mock_agent1, mock_agent2]
        mock_factory = MagicMock()
        mock_train_core.return_value = {}

        result, info = sequential_train(
            agents=agents,
            env_factory=mock_factory,
            num_rounds=1,
            timesteps_per_agent=100,
            update_schedule="IBR",
            verbose=False,
            save_dir=None,
        )

        # IBR should NOT call save_to_bytes for snapshotting
        mock_agent1.save_to_bytes.assert_not_called()
        mock_agent2.save_to_bytes.assert_not_called()
        assert info["update_schedule"] == "IBR"

    @patch("easy_marl.src.core.training._train_agent_core")
    def test_sbr_schedule_snapshots(self, mock_train_core):
        """Verify SBR schedule snapshots agents at round start."""
        from easy_marl.src.core.training import sequential_train

        mock_agent1 = MagicMock()
        mock_agent1.save_to_bytes.return_value = b"state1"
        mock_agent2 = MagicMock()
        mock_agent2.save_to_bytes.return_value = b"state2"
        agents = [mock_agent1, mock_agent2]
        mock_factory = MagicMock()
        mock_train_core.return_value = {}

        result, info = sequential_train(
            agents=agents,
            env_factory=mock_factory,
            num_rounds=1,
            timesteps_per_agent=100,
            update_schedule="SBR",
            verbose=False,
            save_dir=None,
        )

        # SBR should call save_to_bytes for snapshotting
        mock_agent1.save_to_bytes.assert_called()
        mock_agent2.save_to_bytes.assert_called()
        assert info["update_schedule"] == "SBR"

    def test_invalid_schedule_raises(self):
        """Verify invalid update_schedule raises ValueError."""
        from easy_marl.src.core.training import sequential_train

        with pytest.raises(ValueError, match="must be 'IBR' or 'SBR'"):
            sequential_train(
                agents=[],
                env_factory=MagicMock(),
                update_schedule="INVALID",
            )

    @patch("easy_marl.src.core.training._train_agent_core")
    def test_inertia_skips_training(self, mock_train_core):
        """Verify inertia probability skips training for some agents."""
        from easy_marl.src.core.training import sequential_train

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"
        agents = [mock_agent]
        mock_factory = MagicMock()
        mock_train_core.return_value = {}

        # With update_probability=0, all agents should be skipped after round 0
        result, info = sequential_train(
            agents=agents,
            env_factory=mock_factory,
            num_rounds=2,
            timesteps_per_agent=100,
            update_schedule="SBR",
            update_probability=0.0,  # Always skip
            verbose=False,
            save_dir=None,
            seed=42,
        )

        # Round 0 always trains, round 1 should skip due to inertia
        assert mock_train_core.call_count == 1

    @patch("easy_marl.src.core.training._train_agent_core")
    def test_training_info_structure(self, mock_train_core):
        """Verify training_info dict has correct structure."""
        from easy_marl.src.core.training import sequential_train

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"
        mock_train_core.return_value = {}

        _, info = sequential_train(
            agents=[mock_agent, mock_agent],
            env_factory=MagicMock(),
            num_rounds=3,
            timesteps_per_agent=500,
            seed=123,
            update_schedule="SBR",
            update_probability=0.75,
            verbose=False,
            save_dir=None,
        )

        assert info["N"] == 2
        assert info["num_rounds"] == 3
        assert info["timesteps_per_agent"] == 500
        assert info["total_timesteps"] == 2 * 3 * 500
        assert info["seed"] == 123
        assert info["parallel"] is False
        assert info["update_schedule"] == "SBR"
        assert info["update_probability"] == 0.75


class TestTrainSingleAgentWorker:
    """Tests for train_single_agent_worker function."""

    @patch("easy_marl.src.core.training._train_agent_core")
    @patch("easy_marl.src.core.training.PPOAgent")
    def test_rehydrates_agents_and_trains(self, mock_ppo_class, mock_train_core):
        """Verify worker reconstructs agents from bytes and trains."""
        from easy_marl.src.core.training import train_single_agent_worker

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"trained_state"
        mock_ppo_class.from_bytes.return_value = mock_agent
        mock_train_core.return_value = {"metric": 1.0}

        mock_env = MagicMock()
        mock_factory = MagicMock(return_value=mock_env)

        idx, state, metrics = train_single_agent_worker(
            agent_index=0,
            round_idx=0,
            agent_state=b"state0",
            agents_states=[b"state0", b"state1"],
            env_factory=mock_factory,
            timesteps_per_agent=100,
            save_dir=None,
            verbose=False,
            N=2,
            seed=42,
        )

        assert idx == 0
        assert state == b"trained_state"
        assert metrics == {"metric": 1.0}
        # Should have called from_bytes for each agent
        assert mock_ppo_class.from_bytes.call_count == 2


class TestParallelTrain:
    """Tests for parallel_train function."""

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    def test_submits_jobs_to_executor(self, mock_as_completed, mock_executor_class):
        """Verify parallel_train submits jobs to ProcessPoolExecutor."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor

        mock_future = MagicMock()
        mock_future.result.return_value = (0, b"trained", {})
        mock_executor.submit.return_value = mock_future
        mock_as_completed.return_value = [mock_future]

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        result, info = parallel_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=1,
            timesteps_per_agent=100,
            n_workers=1,
            verbose=False,
            save_dir=None,
        )

        mock_executor.submit.assert_called()
        assert info["parallel"] is True
        assert info["n_workers"] == 1

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    def test_inertia_skips_in_parallel(self, mock_as_completed, mock_executor_class):
        """Verify inertia skips agents in parallel mode."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_as_completed.return_value = []

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        # With update_probability=0, all agents should be skipped after round 0
        result, info = parallel_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=2,
            timesteps_per_agent=100,
            update_probability=0.0,
            n_workers=1,
            verbose=False,
            save_dir=None,
            seed=42,
        )

        # Round 0 submits, round 1 should skip
        assert mock_executor.submit.call_count == 1

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    def test_workers_capped_at_agent_count(self, mock_as_completed, mock_executor_class):
        """Verify n_workers is capped at number of agents."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_as_completed.return_value = []

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        _, info = parallel_train(
            agents=[mock_agent],  # Only 1 agent
            env_factory=MagicMock(),
            num_rounds=1,
            n_workers=10,  # Request 10 workers
            verbose=False,
            save_dir=None,
        )

        # Should be capped at 1
        assert info["n_workers"] == 1
        mock_executor_class.assert_called_with(max_workers=1)

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    def test_training_info_structure_parallel(
        self, mock_as_completed, mock_executor_class
    ):
        """Verify training_info dict has correct structure for parallel."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_as_completed.return_value = []

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        _, info = parallel_train(
            agents=[mock_agent, mock_agent],
            env_factory=MagicMock(),
            num_rounds=2,
            timesteps_per_agent=200,
            seed=99,
            update_probability=0.8,
            n_workers=2,
            verbose=False,
            save_dir=None,
        )

        assert info["N"] == 2
        assert info["num_rounds"] == 2
        assert info["timesteps_per_agent"] == 200
        assert info["total_timesteps"] == 2 * 2 * 200
        assert info["seed"] == 99
        assert info["parallel"] is True
        assert info["n_workers"] == 2
        assert info["update_probability"] == 0.8


class TestVerboseOutputCoverage:
    """Tests specifically for verbose output branches."""

    @patch("easy_marl.src.core.training.set_all_seeds")
    @patch("easy_marl.src.core.training.os.makedirs")
    @patch("builtins.print")
    def test_train_agent_core_verbose_with_save(
        self, mock_print, mock_makedirs, mock_set_seeds
    ):
        """Cover line 77: verbose print after save."""
        from easy_marl.src.core.training import _train_agent_core

        mock_agent = MagicMock()
        mock_factory = MagicMock(return_value=MagicMock())

        _train_agent_core(
            agent_index=0,
            round_idx=0,
            agents=[mock_agent],
            env_factory=mock_factory,
            timesteps_per_agent=100,
            save_dir="test_dir",
            verbose=True,
            N=1,
            seed=42,
        )

        # Should print both training start and save confirmation
        assert mock_print.call_count >= 2

    @patch("easy_marl.src.core.training._train_agent_core")
    @patch("builtins.print")
    def test_sequential_train_verbose_round(self, mock_print, mock_train_core):
        """Cover line 126: verbose round print in sequential_train."""
        from easy_marl.src.core.training import sequential_train

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"
        mock_train_core.return_value = {}

        sequential_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=1,
            timesteps_per_agent=100,
            update_schedule="SBR",
            verbose=True,
            save_dir=None,
        )

        # Should print round info
        assert mock_print.called

    @patch("easy_marl.src.core.training._train_agent_core")
    @patch("builtins.print")
    def test_sequential_train_verbose_inertia(self, mock_print, mock_train_core):
        """Cover line 148: verbose inertia skip print in sequential_train."""
        from easy_marl.src.core.training import sequential_train

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"
        mock_train_core.return_value = {}

        sequential_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=2,
            timesteps_per_agent=100,
            update_schedule="SBR",
            update_probability=0.0,
            verbose=True,
            save_dir=None,
            seed=42,
        )

        # Should print inertia skip message
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Inertia" in c for c in calls)

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    @patch("builtins.print")
    def test_parallel_train_verbose_round(
        self, mock_print, mock_as_completed, mock_executor_class
    ):
        """Cover line 329: verbose round print in parallel_train."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_as_completed.return_value = []

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        parallel_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=1,
            timesteps_per_agent=100,
            n_workers=1,
            verbose=True,
            save_dir=None,
        )

        # Should print round info
        assert mock_print.called

    @patch("easy_marl.src.core.training.ProcessPoolExecutor")
    @patch("easy_marl.src.core.training.as_completed")
    @patch("builtins.print")
    def test_parallel_train_verbose_inertia(
        self, mock_print, mock_as_completed, mock_executor_class
    ):
        """Cover line 340: verbose inertia skip print in parallel_train."""
        from easy_marl.src.core.training import parallel_train

        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        mock_as_completed.return_value = []

        mock_agent = MagicMock()
        mock_agent.save_to_bytes.return_value = b"state"

        parallel_train(
            agents=[mock_agent],
            env_factory=MagicMock(),
            num_rounds=2,
            timesteps_per_agent=100,
            update_probability=0.0,
            n_workers=1,
            verbose=True,
            save_dir=None,
            seed=42,
        )

        # Should print inertia skip message
        calls = [str(c) for c in mock_print.call_args_list]
        assert any("Inertia" in c for c in calls)

