from easy_marl.src.core.agents import PPOAgent
from easy_marl.src.core.training import sequential_train, parallel_train

class TestTrainingEquivalence:
    """Tests that sequential and parallel training produce equivalent results."""

    def test_that_one_step_in_sequential_train_is_equivalent_to_one_step_in_parallel_train(
        self,
    ):
        pass