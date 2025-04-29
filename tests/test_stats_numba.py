"""Tests for ensuring code coverage of Numba-accelerated functions in stats.core."""

import numpy as np
from numpy.random import random
from numpy.testing import assert_array_almost_equal

from pastas.stats.core import _compute_ccf_gaussian, _compute_ccf_rectangle


def test_compute_ccf_rectangle_py_func() -> None:
    """Test the Python version of _compute_ccf_rectangle for coverage."""
    # Access the Python version directly via the py_func attribute
    py_func = _compute_ccf_rectangle.py_func

    # Create test data
    lags = np.array([0.0, 1.0, 2.0, 3.0])
    t_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = random(5)
    t_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = random(5)
    bin_width = 0.5

    # Call the Python function directly
    c, b = py_func(lags, t_x, x, t_y, y, bin_width)

    # Simple assertion to ensure the function runs successfully
    assert len(c) == len(lags)
    assert len(b) == len(lags)


def test_compute_ccf_gaussian_py_func() -> None:
    """Test the Python version of _compute_ccf_gaussian for coverage."""
    # Access the Python version directly via the py_func attribute
    py_func = _compute_ccf_gaussian.py_func

    # Create test data
    lags = np.array([0.0, 1.0, 2.0, 3.0])
    t_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x = random(5)
    t_y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = random(5)
    bin_width = 0.5

    # Call the Python function directly
    c, b = py_func(lags, t_x, x, t_y, y, bin_width)

    # Simple assertion to ensure the function runs successfully
    assert len(c) == len(lags)
    assert len(b) == len(lags)


def test_rectangle_vs_gaussian_consistency() -> None:
    """Test that rectangle and gaussian methods produce similar results for simple cases."""
    # Create test data with small differences
    lags = np.array([0.0, 1.0, 2.0])
    t_x = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([1.0, 2.0, 1.0, 0.0])
    t_y = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 1.0])

    # Get Python implementations for both methods
    rect_py = _compute_ccf_rectangle.py_func
    gauss_py = _compute_ccf_gaussian.py_func

    # Calculate with both methods using narrow bins
    c_rect, _ = rect_py(lags, t_x, x, t_y, y, bin_width=0.1)
    c_gauss, _ = gauss_py(lags, t_x, x, t_y, y, bin_width=0.1)

    # With narrow bins, methods should be similar for simple cases
    # (not exactly equal due to different weighting)
    assert_array_almost_equal(c_rect, c_gauss, decimal=1)
