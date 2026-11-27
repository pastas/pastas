"""Tests for ensuring code coverage of Numba-accelerated functions in stats.core."""

import numpy as np
import pandas as pd
from numpy.random import random
from numpy.testing import assert_allclose, assert_array_almost_equal

import pastas as ps
from pastas.stats.core import (
    _compute_ccf_gaussian,
    _compute_ccf_gaussian_numpy,
    _compute_ccf_rectangle,
    _compute_ccf_rectangle_numpy,
    _preprocess,
)


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


def test_ccf_numba_and_numpy_match_exact() -> None:
    """Verify Numba two-pointer and NumPy fallback match naive reference exactly."""
    rng = np.random.default_rng(123)
    n = 150
    t_x = np.cumsum(rng.uniform(0.4, 2.8, size=n))
    t_y = np.cumsum(rng.uniform(0.3, 2.9, size=n))
    x = rng.normal(0, 1, size=n)
    y = rng.normal(0, 1, size=n)
    lags = np.array([0.0, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 100.0])
    bin_width = 0.5

    # Reference naive O(lags * N^2) implementation
    def naive_rect(lags_arr, tx, xv, ty, yv, bw):
        c_ref = np.empty_like(lags_arr)
        b_ref = np.empty_like(lags_arr)
        for k, lag_k in enumerate(lags_arr):
            cl = 0.0
            bs = 0.0
            for j in range(len(tx)):
                for i in range(j, len(tx)):
                    if abs(abs(tx[i] - ty[j]) - lag_k) <= bw:
                        cl += xv[i] * yv[j]
                        bs += 1.0
            c_ref[k] = np.nan if bs == 0.0 else cl / bs
            b_ref[k] = 1e-16 if bs == 0.0 else bs
        return c_ref, b_ref

    def naive_gauss(lags_arr, tx, xv, ty, yv, bw):
        c_ref = np.empty_like(lags_arr)
        b_ref = np.empty_like(lags_arr)
        den1 = -2.0 * bw**2
        den2 = np.sqrt(2.0 * np.pi * bw)
        six_den2 = 6.0 * den2
        for k, lag_k in enumerate(lags_arr):
            cl = 0.0
            bs = 0.0
            for j in range(len(tx)):
                for i in range(j, len(tx)):
                    dtlag = tx[i] - ty[j] - lag_k
                    if abs(dtlag) < six_den2:
                        d = np.exp(dtlag**2 / den1) / den2
                        cl += xv[i] * yv[j] * d
                        bs += d
            c_ref[k] = np.nan if bs == 0.0 else cl / bs
            b_ref[k] = 1e-16 if bs == 0.0 else bs
        return c_ref, b_ref

    # Both ACF (t_x == t_y) and general CCF (t_x != t_y)
    for tx, ty in [(t_x, t_x), (t_x, t_y)]:
        c_r_ref, b_r_ref = naive_rect(lags, tx, x, ty, y, bin_width)
        c_r_nb, b_r_nb = _compute_ccf_rectangle(lags, tx, x, ty, y, bin_width)
        c_r_np, b_r_np = _compute_ccf_rectangle_numpy(lags, tx, x, ty, y, bin_width)

        assert_allclose(c_r_nb, c_r_ref, rtol=1e-13, atol=1e-13, equal_nan=True)
        assert_allclose(b_r_nb, b_r_ref, rtol=1e-13, atol=1e-13)
        assert_allclose(c_r_np, c_r_ref, rtol=1e-13, atol=1e-13, equal_nan=True)
        assert_allclose(b_r_np, b_r_ref, rtol=1e-13, atol=1e-13)

        c_g_ref, b_g_ref = naive_gauss(lags, tx, x, ty, y, bin_width)
        c_g_nb, b_g_nb = _compute_ccf_gaussian(lags, tx, x, ty, y, bin_width)
        c_g_np, b_g_np = _compute_ccf_gaussian_numpy(lags, tx, x, ty, y, bin_width)

        assert_allclose(c_g_nb, c_g_ref, rtol=1e-13, atol=1e-13, equal_nan=True)
        assert_allclose(b_g_nb, b_g_ref, rtol=1e-13, atol=1e-13)
        assert_allclose(c_g_np, c_g_ref, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert_allclose(b_g_np, b_g_ref, rtol=1e-10, atol=1e-10)


def test_acf_options_numba_toggle_parity() -> None:
    """Verify ps.stats.acf matches with ps.options.numba True vs False."""
    rng = np.random.default_rng(99)
    idx = pd.Timestamp("2020-01-01") + pd.to_timedelta(
        np.cumsum(rng.integers(12, 60, size=250)), unit="h"
    )
    ser = pd.Series(np.cumsum(rng.normal(0, 1, size=250)), index=idx)

    orig_opt = ps.options.numba
    try:
        for method in ("gaussian", "rectangle"):
            ps.options.numba = True
            r_nb = ps.stats.acf(ser, lags=90, bin_method=method, full_output=True)
            ps.options.numba = False
            r_np = ps.stats.acf(ser, lags=90, bin_method=method, full_output=True)

            assert r_nb.index.equals(r_np.index)
            assert_allclose(r_nb["acf"].values, r_np["acf"].values, atol=1e-12)
            assert_allclose(r_nb["conf"].values, r_np["conf"].values, atol=1e-12)
            assert_allclose(r_nb["n"].values, r_np["n"].values, rtol=1e-10, atol=1e-10)
    finally:
        ps.options.numba = orig_opt


def test_ccf_unsorted_timestamps_fallback() -> None:
    """Verify unsorted timestamps safely fall back without wrong results."""
    lags = np.array([0.0, 1.0, 2.0])
    t_x = np.array([2.0, 0.0, 3.0, 1.0])
    t_y = np.array([2.0, 0.0, 3.0, 1.0])
    x = np.array([1.0, -1.0, 0.5, 0.2])
    y = x.copy()

    c_r_nb, b_r_nb = _compute_ccf_rectangle.py_func(lags, t_x, x, t_y, y, 0.5)
    c_r_np, b_r_np = _compute_ccf_rectangle_numpy(lags, t_x, x, t_y, y, 0.5)
    assert_allclose(c_r_nb, c_r_np, equal_nan=True)
    assert_allclose(b_r_nb, b_r_np)

    c_g_nb, b_g_nb = _compute_ccf_gaussian.py_func(lags, t_x, x, t_y, y, 0.5)
    c_g_np, b_g_np = _compute_ccf_gaussian_numpy(lags, t_x, x, t_y, y, 0.5)
    assert_allclose(c_g_nb, c_g_np, equal_nan=True)
    assert_allclose(b_g_nb, b_g_np)


def test_preprocess_preserves_all_timestamps_and_alignment() -> None:
    """Verify _preprocess keeps all observations aligned with their true timestamps."""
    idx = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-12", "2020-01-21"])
    ser = pd.Series([1.0, 2.0, 2.0, 5.0], index=idx)

    x, t, _ = _preprocess(ser)
    assert len(t) == len(x) == len(ser)
    assert_allclose(t, np.array([0.0, 10.0, 11.0, 20.0]))

    # With bin_method="rectangle", lag=1 day pairs index 1 (day 10) & index 2 (day 11),
    # and lag=20 days pairs index 0 (day 0) & index 3 (day 20, the last observation).
    res = ps.stats.acf(
        ser,
        lags=np.array([1.0, 20.0]),
        bin_method="rectangle",
        min_obs=0,
        full_output=True,
    )
    assert res.loc[pd.Timedelta(days=1), "n"] == 1.0
    assert res.loc[pd.Timedelta(days=20), "n"] == 1.0
    assert_allclose(res.loc[pd.Timedelta(days=1), "acf"], x[1] * x[2])
    assert_allclose(res.loc[pd.Timedelta(days=20), "acf"], x[0] * x[3])
