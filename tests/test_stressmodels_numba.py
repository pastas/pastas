"""Tests for numba-accelerated functions in stressmodels.py."""

import numpy as np

from pastas.stressmodels import TarsoModel


class TestTarsoModelNumba:
    """Test the numba-accelerated tarso function in TarsoModel."""

    def setup_method(self) -> None:
        """Setup for tests."""
        # Create test data for recharge input
        np.random.seed(42)  # For reproducibility
        self.npoints = 100
        self.r = np.random.normal(1, 0.5, self.npoints)  # Simple recharge series

    def test_tarso_py_func(self) -> None:
        """Test the Python implementation of the tarso function."""
        # Access the Python version directly via py_func attribute
        tarso_py_func = TarsoModel.tarso.py_func

        # Parameters (A0, a0, d0, A1, a1, d1)
        p = np.array([0.9, 10.0, 4.0, 0.8, 15.0, 6.0])
        dt = 1.0

        # Call the Python function directly
        result = tarso_py_func(p, self.r, dt)

        # Check basic properties
        assert len(result) == len(self.r)
        assert not np.isnan(result).any()

        # Result should be bounded by drainage levels plus some margin
        assert np.all(result >= p[2] - 1)  # Near or above d0
        assert np.all(result <= p[5] + 1)  # Near or below d1

    def test_tarso_parameter_variations(self) -> None:
        """Test tarso function with different parameter values."""
        tarso_py_func = TarsoModel.tarso.py_func
        dt = 1.0

        # Test different parameter combinations
        test_parameters = [
            # Different drainage level scenarios
            np.array([0.8, 10.0, 2.0, 0.7, 12.0, 4.0]),  # Lower drainage levels
            np.array([0.9, 15.0, 6.0, 0.8, 20.0, 8.0]),  # Higher drainage levels
            np.array([0.9, 10.0, 4.0, 0.8, 15.0, 4.5]),  # Close drainage levels
            # Different response characteristics
            np.array([0.5, 5.0, 4.0, 0.4, 7.0, 6.0]),  # Faster response
            np.array([1.5, 25.0, 4.0, 1.3, 30.0, 6.0]),  # Slower response
        ]

        for p in test_parameters:
            result = tarso_py_func(p, self.r, dt)

            # Basic checks for each parameter set
            assert len(result) == len(self.r)
            assert not np.isnan(result).any()

            # Results should generally be bounded by drainage levels
            assert np.all(result >= p[2] - 2)
            assert np.all(result <= p[5] + 2)

    def test_tarso_dt_variations(self) -> None:
        """Test tarso function with different time step values."""
        tarso_py_func = TarsoModel.tarso.py_func
        p = np.array([0.9, 10.0, 4.0, 0.8, 15.0, 6.0])

        # Test different dt values
        for dt in [0.1, 0.5, 1.0, 2.0, 5.0]:
            result = tarso_py_func(p, self.r, dt)

            # Basic checks
            assert len(result) == len(self.r)
            assert not np.isnan(result).any()
