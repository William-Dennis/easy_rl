import pytest
import numpy as np
from easy_marl.src.observators import OBSERVERS

# Define constants for the test
N_AGENTS = 5
T_STEPS = 24

# Create dummy data using numpy for array inputs
DUMMY_INPUTS = {
    "D_profile": np.zeros(T_STEPS),
    "K": np.ones(N_AGENTS) * 10,
    "c": np.ones(N_AGENTS) * 5,
    "agent_index": 0,
    "demand_scale": 1.0,
    "capacity_scale": 1.0,
    "cost_scale": 1.0,
    "mask": np.array([False, True, True, True, True]),  # Mask for other agents
    "t": 0,
}


class TestObservatorsDimensionsMatch:
    """Test that the observer dimensions match the expected sizes."""

    @pytest.mark.parametrize("observer_name", list(OBSERVERS.keys()))
    def test_observer_dimension(self, observer_name):
        """Test that the observer dimension function matches the actual observer output size."""
        dim_function, obs_function = OBSERVERS[observer_name]

        # 1. Get expected dimension
        expected_dim = dim_function(N_AGENTS)

        # 2. Create a buffer of that size
        obs_buf = np.zeros(expected_dim)

        # 3. Run observer function with dummy inputs
        # We add obs_buf to the inputs dynamically
        inputs = DUMMY_INPUTS.copy()
        inputs["obs_buf"] = obs_buf

        # Call the function unpacking the inputs
        result_buf = obs_function(**inputs)

        # 4. Check that dimensions match
        assert len(result_buf) == expected_dim, (
            f"Dimension mismatch for observer '{observer_name}': "
            f"Expected {expected_dim}, got {len(result_buf)}"
        )
