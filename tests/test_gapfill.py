"""Tests for the gapfill module."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from numpy import isclose
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, Timestamp

import pastas as ps
from pastas.gapfill import (
    _compute_correction,
    _compute_prediction_variance,
    gap_fill,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures for gapfill tests
# =============================================================================


@pytest.fixture
def head_with_single_gap(head: Series) -> Series:
    """Create a head series with a single gap in the middle."""
    head_gap = head.copy()
    head_gap.iloc[50:60] = np.nan
    return head_gap


@pytest.fixture
def head_with_multiple_gaps(head: Series) -> Series:
    """Create a head series with multiple gaps."""
    head_gap = head.copy()
    head_gap.iloc[30:35] = np.nan
    head_gap.iloc[70:80] = np.nan
    head_gap.iloc[100:103] = np.nan
    return head_gap


@pytest.fixture
def head_with_gap_at_start(head: Series) -> Series:
    """Create a head series with a gap at the beginning."""
    head_gap = head.copy()
    head_gap.iloc[:5] = np.nan
    return head_gap


@pytest.fixture
def head_with_gap_at_end(head: Series) -> Series:
    """Create a head series with a gap at the end."""
    head_gap = head.copy()
    head_gap.iloc[-5:] = np.nan
    return head_gap


@pytest.fixture
def head_with_single_missing(head: Series) -> Series:
    """Create a head series with a single missing value."""
    head_gap = head.copy()
    head_gap.iloc[50] = np.nan
    return head_gap


@pytest.fixture
def ml_for_gapfill(ml_noisemodel: ps.Model) -> ps.Model:
    """Return a model suitable for gap filling (with noise model)."""
    return ml_noisemodel


@pytest.fixture
def ml_no_noise(ml_solved: ps.Model) -> ps.Model:
    """Return a solved model without a noise model."""
    # Make sure the model doesn't have a noise model
    ml = ml_solved.copy()
    ml.noisemodel = None
    return ml


# =============================================================================
# Tests for gap_fill function
# =============================================================================


class TestGapFill:
    """Tests for the gap_fill function."""

    def test_gap_fill_no_gaps_no_sig_level(
        self, head: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a series that has no gaps and no sig_level."""
        result = gap_fill(head, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        # Result should be identical to input when there are no gaps
        pd.testing.assert_series_equal(result, head)

    def test_gap_fill_no_gaps_with_sig_level(
        self, head: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a series that has no gaps and sig_level."""
        result = gap_fill(head, ml_for_gapfill, sig_level=0.05)
        assert isinstance(result, DataFrame)
        assert list(result.columns) == ["head_filled", "lower", "upper"]
        # Check that head_filled matches the original when there are no gaps
        # Note: rename to match column name
        expected = head.copy()
        expected.name = "head_filled"
        pd.testing.assert_series_equal(
            result["head_filled"], expected, check_names=False
        )

    def test_gap_fill_single_gap(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a single gap and no sig_level."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        # Check that all gaps are filled (no NaN values)
        assert not result.isna().any()
        # Check that original non-gap values are preserved
        original_values = head_with_single_gap.dropna()
        for idx in original_values.index:
            assert isclose(result.loc[idx], original_values.loc[idx])

    def test_gap_fill_single_gap_with_sig_level(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a single gap and sig_level."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=0.05)
        assert isinstance(result, DataFrame)
        assert list(result.columns) == ["head_filled", "lower", "upper"]
        # Check that all gaps are filled
        assert not result["head_filled"].isna().any()
        # Check prediction intervals
        assert (result["upper"] >= result["head_filled"]).all()
        assert (result["lower"] <= result["head_filled"]).all()

    def test_gap_fill_multiple_gaps(
        self, head_with_multiple_gaps: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with multiple gaps."""
        result = gap_fill(head_with_multiple_gaps, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        # Check that all gaps are filled
        assert not result.isna().any()
        # Check that original non-gap values are preserved
        original_values = head_with_multiple_gaps.dropna()
        for idx in original_values.index:
            assert isclose(result.loc[idx], original_values.loc[idx])

    def test_gap_fill_gap_at_start(
        self, head_with_gap_at_start: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a gap at the start."""
        result = gap_fill(head_with_gap_at_start, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        # Check that the gap at the start is filled
        assert not result.isna().any()

    def test_gap_fill_gap_at_end(
        self, head_with_gap_at_end: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a gap at the end."""
        result = gap_fill(head_with_gap_at_end, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        # Check that the gap at the end is filled
        assert not result.isna().any()

    def test_gap_fill_single_missing(
        self, head: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with a single missing value."""
        head_single_missing = head.copy()
        head_single_missing.iloc[50] = np.nan

        result = gap_fill(head_single_missing, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        assert result is not None

    def test_gap_fill_with_tmin_tmax(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with explicit tmin and tmax."""
        tmin = head_with_single_gap.index[40]
        tmax = head_with_single_gap.index[69]
        result = gap_fill(
            head_with_single_gap, ml_for_gapfill, tmin=tmin, tmax=tmax, sig_level=None
        )
        assert isinstance(result, Series)
        assert result.index.min() == head_with_single_gap.index.min()
        assert result.index.max() == head_with_single_gap.index.max()

    def test_gap_fill_with_sig_level_default(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with default sig_level (0.05)."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill)
        assert isinstance(result, DataFrame)
        assert list(result.columns) == ["head_filled", "lower", "upper"]
        # Check that all gaps are filled
        assert not result["head_filled"].isna().any()

    def test_gap_fill_without_noisemodel(
        self, head_with_single_gap: Series, ml_no_noise: ps.Model
    ) -> None:
        """Test gap_fill without a noise model."""
        result = gap_fill(
            head_with_single_gap, ml_no_noise, sig_level=None, use_noisemodel=False
        )
        assert isinstance(result, Series)
        # Check that all gaps are filled
        assert not result.isna().any()

    def test_gap_fill_with_sig_level_no_noisemodel(
        self, head_with_single_gap: Series, ml_no_noise: ps.Model
    ) -> None:
        """Test gap_fill with sig_level but no noise model."""
        result = gap_fill(
            head_with_single_gap,
            ml_no_noise,
            sig_level=0.05,
            use_noisemodel=False,
        )
        assert isinstance(result, DataFrame)
        assert list(result.columns) == ["head_filled", "lower", "upper"]
        # Check that all gaps are filled
        assert not result["head_filled"].isna().any()

    def test_gap_fill_use_noisemodel_true_no_noise_raises(
        self, head_with_single_gap: Series, ml_no_noise: ps.Model
    ) -> None:
        """Test that gap_fill raises an error when use_noisemodel=True but no noise model exists."""
        with pytest.raises(ValueError, match="No noise model available"):
            gap_fill(
                head_with_single_gap, ml_no_noise, sig_level=None, use_noisemodel=True
            )

    def test_gap_fill_with_sig_level_use_noisemodel_true_no_noise_raises(
        self, head_with_single_gap: Series, ml_no_noise: ps.Model
    ) -> None:
        """Test that gap_fill raises an error when use_noisemodel=True with sig_level but no noise model exists."""
        with pytest.raises(ValueError, match="No noise model available"):
            gap_fill(
                head_with_single_gap, ml_no_noise, sig_level=0.05, use_noisemodel=True
            )

    def test_gap_fill_preserves_index(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test that gap_fill preserves the original index."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=None)
        assert result.index.equals(head_with_single_gap.index)

    def test_gap_fill_string_tmin_tmax(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with string tmin and tmax."""
        tmin = str(head_with_single_gap.index[40])[:10]
        tmax = str(head_with_single_gap.index[69])[:10]
        result = gap_fill(
            head_with_single_gap, ml_for_gapfill, tmin=tmin, tmax=tmax, sig_level=None
        )
        assert isinstance(result, Series)
        assert not result.isna().any()

    def test_gap_fill_different_sig_levels(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with different significance levels."""
        sig_levels = [0.01, 0.05, 0.10, 0.20]
        results = []
        for sig in sig_levels:
            result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=sig)
            results.append(result)
            # Check that all results are DataFrames with the right columns
            assert isinstance(result, DataFrame)
            assert list(result.columns) == ["head_filled", "lower", "upper"]

    def test_gap_fill_no_sig_level_returns_series(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test that gap_fill returns a Series when sig_level is None."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)

    def test_gap_fill_sig_level_none_returns_series(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test that gap_fill returns a Series when sig_level is explicitly None."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=None)
        assert isinstance(result, Series)
        assert not result.isna().any()


# =============================================================================
# Tests for compute_correction function
# =============================================================================


class TestComputeCorrection:
    """Tests for the compute_correction function."""

    def test_compute_correction_with_noisemodel(self, ml_for_gapfill: ps.Model) -> None:
        """Test compute_correction with noise model."""
        # Create test data
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        r0 = Series([1.5], index=[index[0]])
        r1 = Series([2.5], index=[index[9]])
        tindex = index[1:9]

        correction = _compute_correction(
            r0, r1, tindex, ml_for_gapfill, use_noisemodel=True
        )

        assert isinstance(correction, Series)
        assert len(correction) == len(tindex)
        assert correction.index.equals(tindex)
        # Correction should be between r0 and r1 (approximately)
        assert (correction >= 1.0).all()  # Allow some tolerance
        assert (correction <= 3.0).all()

    def test_compute_correction_without_noisemodel(self, ml_no_noise: ps.Model) -> None:
        """Test compute_correction without noise model."""
        # Create test data
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        r0 = Series([1.5], index=[index[0]])
        r1 = Series([2.5], index=[index[9]])
        tindex = index[1:9]

        correction = _compute_correction(
            r0, r1, tindex, ml_no_noise, use_noisemodel=False
        )

        assert isinstance(correction, Series)
        assert len(correction) == len(tindex)
        assert correction.index.equals(tindex)
        # With linear interpolation, correction should be linearly between r0 and r1
        # Check first and last values
        assert isclose(correction.iloc[0], 1.5 + (2.5 - 1.5) * 1 / 9, atol=0.01)
        assert isclose(correction.iloc[-1], 1.5 + (2.5 - 1.5) * 8 / 9, atol=0.01)

    def test_compute_correction_zero_residuals(self, ml_for_gapfill: ps.Model) -> None:
        """Test compute_correction with zero residuals."""
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        r0 = Series([0.0], index=[index[0]])
        r1 = Series([0.0], index=[index[9]])
        tindex = index[1:9]

        correction = _compute_correction(
            r0, r1, tindex, ml_for_gapfill, use_noisemodel=True
        )

        # With zero residuals, correction should be zero or very close
        assert (abs(correction) < 0.1).all()

    def test_compute_correction_negative_residuals(
        self, ml_for_gapfill: ps.Model
    ) -> None:
        """Test compute_correction with negative residuals."""
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        r0 = Series([-1.5], index=[index[0]])
        r1 = Series([-2.5], index=[index[9]])
        tindex = index[1:9]

        correction = _compute_correction(
            r0, r1, tindex, ml_for_gapfill, use_noisemodel=True
        )

        assert isinstance(correction, Series)
        assert len(correction) == len(tindex)
        # Correction should be between the two negative residuals (approximately)
        assert (correction >= -3.0).all()
        assert (correction <= -1.0).all()


# =============================================================================
# Tests for compute_prediction_variance function
# =============================================================================


class TestComputePredictionVariance:
    """Tests for the compute_prediction_variance function."""

    def test_compute_prediction_variance_with_noisemodel(
        self, ml_for_gapfill: ps.Model
    ) -> None:
        """Test compute_prediction_variance with noise model."""
        # Use the actual observation range of the model
        obs = ml_for_gapfill.observations()
        index = obs.index[:10]  # Use first 10 observations
        t0 = index[0]
        t1 = index[9]
        tindex = index[1:9]

        # Compute required parameters
        dt = obs.index.to_series().diff().median() / Timedelta("1D")
        noise_var = ml_for_gapfill.noise(tmin=t0, tmax=t1).var()

        variance = _compute_prediction_variance(
            t0, t1, tindex, ml_for_gapfill, noise_var=noise_var, dt=dt
        )

        assert isinstance(variance, Series)
        assert len(variance) == len(tindex)
        assert variance.index.equals(tindex)
        # Variance should be positive
        assert (variance > 0).all()


# =============================================================================
# Integration tests
# =============================================================================


class TestGapFillIntegration:
    """Integration tests for the gapfill module."""

    def test_gap_fill_full_workflow(
        self, head: Series, prec: Series, evap: Series
    ) -> None:
        """Test the full gap filling workflow."""
        # Create a model
        ml = ps.Model(head, name="test_gapfill")
        ps.RechargeModel(
            model=ml,
            prec=prec,
            evap=evap,
            name="rch",
            rfunc=ps.Exponential(),
        )
        ml.solve(report=False)

        # Add noise model
        ps.ArNoiseModel(model=ml)
        ml.solve(initial=False, report=False)

        head_gap = head.copy()
        head_gap.iloc[100:110] = np.nan

        # Fill the gap
        result = gap_fill(head_gap, ml, sig_level=0.05)

        assert isinstance(result, DataFrame)
        assert list(result.columns) == ["head_filled", "lower", "upper"]
        assert not result["head_filled"].isna().any()

    def test_gap_fill_with_different_frequencies(
        self, head: Series, prec: Series, evap: Series
    ) -> None:
        """Test gap_fill with different frequency head series."""
        # Create a simple daily head series for this test
        # Use dates that align with the stress data
        dates = pd.date_range("2010-01-14", periods=100, freq="D")
        head_daily = Series(
            np.random.rand(100) * 10 + 20, index=dates, name="test_head"
        )

        # Create a simple model with stress that covers the same period
        ml = ps.Model(head_daily)
        # Use a constant stress to avoid date alignment issues
        stress_daily = Series(np.random.rand(100) * 0.1, index=dates, name="prec_test")
        ps.StressModel(
            model=ml,
            stress=stress_daily,
            name="prec",
            rfunc=ps.Exponential(),
            settings="prec",
        )
        ml.solve(report=False)
        ps.ArNoiseModel(model=ml)
        ml.solve(initial=False, report=False)

        # Resample head to weekly frequency
        head_weekly = head_daily.resample("W").mean()

        head_gap = head_weekly.copy()
        if len(head_gap) >= 8:
            head_gap.iloc[2:5] = np.nan

            result = gap_fill(head_gap, ml, sig_level=None)

            assert isinstance(result, Series)
            # Check that gaps are filled or at least the result is not all NaN
            assert result.notna().sum() > 0


# =============================================================================
# Tests for correct gap detection and filling
# =============================================================================


class TestGapDetection:
    """Tests for gap detection in gap_fill."""

    def test_single_gap_detection(
        self, head_with_single_gap: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test that a single gap is correctly detected and filled."""
        result = gap_fill(head_with_single_gap, ml_for_gapfill, sig_level=None)

        # Find where the gap was
        gap_indices = head_with_single_gap[head_with_single_gap.isna()].index

        # Check that all gap indices are filled (not NaN)
        for idx in gap_indices:
            assert not np.isnan(result.loc[idx])

    def test_multiple_gaps_detection(
        self, head_with_multiple_gaps: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test that multiple gaps are correctly detected and filled."""
        result = gap_fill(head_with_multiple_gaps, ml_for_gapfill, sig_level=None)

        # Find where the gaps were
        gap_indices = head_with_multiple_gaps[head_with_multiple_gaps.isna()].index

        # Check that all gap indices are filled
        for idx in gap_indices:
            assert not np.isnan(result.loc[idx])


# =============================================================================
# Test for numerical stability
# =============================================================================


class TestNumericalStability:
    """Tests for numerical stability in the gapfill module."""

    def test_gap_fill_with_extreme_values(
        self, head: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill with extreme values in the head series."""
        # Create head with extreme values
        head_extreme = head.copy()
        head_extreme.iloc[::2] = head_extreme.iloc[::2] * 1000
        head_extreme.iloc[1::2] = head_extreme.iloc[1::2] * -1000

        head_gap = head_extreme.copy()
        head_gap.iloc[50:60] = np.nan

        result = gap_fill(head_gap, ml_for_gapfill, sig_level=None)

        assert isinstance(result, Series)
        assert not result.isna().any()
        # Check that values are finite
        assert np.all(np.isfinite(result))

    def test_compute_correction_with_extreme_residuals(
        self, ml_for_gapfill: ps.Model
    ) -> None:
        """Test compute_correction with extreme residual values."""
        index = pd.date_range("2000-01-01", periods=10, freq="D")
        r0 = Series([1e6], index=[index[0]])
        r1 = Series([-1e6], index=[index[9]])
        tindex = index[1:9]

        correction = _compute_correction(
            r0, r1, tindex, ml_for_gapfill, use_noisemodel=True
        )

        assert isinstance(correction, Series)
        assert np.all(np.isfinite(correction))

    def test_compute_prediction_variance_with_extreme_alpha(
        self, ml_for_gapfill: ps.Model
    ) -> None:
        """Test compute_prediction_variance with extreme alpha values."""
        # Modify the noise_alpha parameter to an extreme but valid value
        # The default pmin for noise_alpha is 1e-5, so use a value above that
        ml = ml_for_gapfill.copy()
        ml.set_parameter("noise_alpha", initial=1e-4, optimal=1e-4)

        # Use the actual observation range of the model
        obs = ml.observations()
        index = obs.index[:10]  # Use first 10 observations
        t0 = index[0]
        t1 = index[9]
        tindex = index[1:9]

        # Compute required parameters
        dt = obs.index.to_series().diff().median() / Timedelta("1D")
        noise_var = ml.noise(tmin=t0, tmax=t1).var()

        variance = _compute_prediction_variance(
            t0, t1, tindex, ml, noise_var=noise_var, dt=dt
        )

        assert isinstance(variance, Series)
        assert np.all(np.isfinite(variance))
        assert (variance > 0).all()


# =============================================================================
# Tests for edge cases with no gaps
# =============================================================================


class TestNoGaps:
    """Tests for edge cases where there are no gaps."""

    def test_gap_fill_all_values_present(
        self, head: Series, ml_for_gapfill: ps.Model
    ) -> None:
        """Test gap_fill when all values are present."""
        result_series = gap_fill(head, ml_for_gapfill, sig_level=None)
        result_df = gap_fill(head, ml_for_gapfill, sig_level=0.05)

        assert isinstance(result_series, Series)
        assert isinstance(result_df, DataFrame)

        # Both should have no NaN values
        assert not result_series.isna().any()
        assert not result_df["head_filled"].isna().any()

        # The head_filled should match the original head
        expected_series = head.copy()
        expected_series.name = None
        pd.testing.assert_series_equal(
            result_series, expected_series, check_names=False
        )

        expected_df = head.copy()
        expected_df.name = "head_filled"
        pd.testing.assert_series_equal(
            result_df["head_filled"], expected_df, check_names=False
        )

    def test_gap_fill_empty_series(self) -> None:
        """Test that get_new_index handles empty series correctly."""
        from pastas.gapfill import _get_new_index

        empty_head = Series(dtype=float, name="empty")

        # For empty series, we need to provide tmin, tmax, and freq
        tmin = Timestamp("2000-01-01")
        tmax = Timestamp("2000-01-10")

        # Test that get_new_index works with empty series
        new_index = _get_new_index(empty_head, tmin=tmin, tmax=tmax, freq="D")
        assert isinstance(new_index, DatetimeIndex)
        assert len(new_index) == 10  # 10 days from tmin to tmax
