import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import pastas as ps
from pastas.stats.dutch import (
    _in_spring,
    _q_gxg,
    gvg,
    q_ghg,
    q_glg,
    q_gvg,
)


@pytest.fixture
def sample_timeseries() -> pd.Series:
    """Create a sample timeseries for testing."""
    # Create daily data spanning multiple years
    index = pd.date_range(start="2017-01-01", end="2021-12-31", freq="D")
    # Create seasonal pattern with some noise
    time = np.arange(len(index))
    seasonal = 10 * np.sin(2 * np.pi * time / 365.25) + 100
    noise = np.random.normal(0, 1, len(index))
    values = seasonal + noise

    # Add some missing data
    series = pd.Series(values, index=index)
    mask = np.random.random(len(series)) < 0.1
    series = series.mask(mask)

    return series


@pytest.fixture
def biweekly_timeseries() -> pd.Series:
    """Create a biweekly timeseries for testing."""
    # Create data on the 14th and 28th of each month
    dates = []
    for year in range(2017, 2021):
        for month in range(1, 13):
            dates.append(pd.Timestamp(year=year, month=month, day=14))
            if month == 2 and year % 4 != 0:
                # Skip Feb 28 in non-leap years
                continue
            if month == 2 and day_exists(year, month, 28):
                dates.append(pd.Timestamp(year=year, month=month, day=28))
            elif month != 2:
                dates.append(pd.Timestamp(year=year, month=month, day=28))

    # Create seasonal pattern with some noise
    time = np.arange(len(dates))
    seasonal = 10 * np.sin(2 * np.pi * time / 24) + 100  # 24 measurements per year
    noise = np.random.normal(0, 1, len(dates))
    values = seasonal + noise

    return pd.Series(values, index=pd.DatetimeIndex(dates))


def day_exists(year: int, month: int, day: int) -> bool:
    """Check if a day exists in a given month and year."""
    try:
        pd.Timestamp(year=year, month=month, day=day)
        return True
    except ValueError:
        return False


def test_in_spring() -> None:
    """Test the _in_spring helper function."""
    # Create test series with dates inside and outside spring
    index = pd.DatetimeIndex(
        [
            "2020-03-01",
            "2020-03-14",
            "2020-03-15",
            "2020-03-31",
            "2020-04-01",
            "2020-04-14",
            "2020-04-15",
            "2020-04-30",
        ]
    )
    series = pd.Series(np.arange(len(index)), index=index)

    # Expected results - True for dates between March 14 and April 14 (inclusive)
    expected = pd.Series(
        [False, True, True, True, True, True, False, False], index=index
    )

    # Test the function
    result = _in_spring(series)
    assert_series_equal(result, expected)


def test_q_gxg(sample_timeseries: pd.Series) -> None:
    """Test the _q_gxg helper function."""

    # Test q_gxg with different quantiles
    result_high = _q_gxg(sample_timeseries, q=0.94, by_year=True)
    result_low = _q_gxg(sample_timeseries, q=0.06, by_year=True)

    # High quantile should be higher than low quantile
    assert result_high > result_low

    # Test by_year parameter
    result_by_year = _q_gxg(sample_timeseries, q=0.5, by_year=True)
    result_all = _q_gxg(sample_timeseries, q=0.5, by_year=False)

    # Results should be roughly similar but not identical
    assert abs(result_by_year - result_all) < 1.0
    assert result_by_year != result_all


def test_q_ghg(sample_timeseries: pd.Series) -> None:
    """Test q_ghg function."""
    # Test default parameters
    result = q_ghg(sample_timeseries)
    assert isinstance(result, float)

    # Test with date range
    tmin = pd.Timestamp("2018-01-01")
    tmax = pd.Timestamp("2019-12-31")
    result_range = q_ghg(sample_timeseries, tmin=tmin, tmax=tmax)
    assert isinstance(result_range, float)

    # Test by_year parameter
    result_all = q_ghg(sample_timeseries, by_year=False)
    assert isinstance(result_all, float)


def test_q_glg(sample_timeseries: pd.Series) -> None:
    """Test q_glg function."""
    # Test default parameters
    result = q_glg(sample_timeseries)
    assert isinstance(result, float)

    # Ensure q_glg gives lower values than q_ghg
    assert q_glg(sample_timeseries) < q_ghg(sample_timeseries)


def test_q_gvg(sample_timeseries: pd.Series) -> None:
    """Test q_gvg function."""
    # Test default parameters
    result = q_gvg(sample_timeseries)
    assert isinstance(result, float)

    # Test with and without by_year
    result1 = q_gvg(sample_timeseries, by_year=True)
    result2 = q_gvg(sample_timeseries, by_year=False)
    assert result1 != result2


def test_gvg(biweekly_timeseries: pd.Series) -> None:
    """Test gvg function."""
    # Test default parameters
    result = gvg(biweekly_timeseries)
    assert isinstance(result, float)

    # Test with different limit values
    result1 = gvg(biweekly_timeseries, limit=4)
    result2 = gvg(biweekly_timeseries, limit=12)
    assert np.isnan(result1) or np.isnan(result2) or result1 != result2

    # Test with None limit (no filling)
    result_none = gvg(biweekly_timeseries, limit=None)
    assert isinstance(result_none, float)


def test_get_spring(sample_timeseries: pd.Series) -> None:
    """Test the _get_spring helper function."""
    from pastas.stats.dutch import _get_spring

    # Extract spring values
    spring_values = _get_spring(sample_timeseries, min_n_meas=2)

    # Verify only spring values are returned
    for date in spring_values.index:
        assert (date.month == 3 and date.day >= 14) or (
            date.month == 4 and date.day < 15
        )

    # Test with insufficient measurements
    sparse_series = sample_timeseries.iloc[::10]  # Take every 10th value
    result = _get_spring(sparse_series, min_n_meas=5)
    # Check if there are enough measurements in result
    if len(result) < 5:  # If not enough measurements
        assert len(result) == 0
    else:
        # If we have values, make sure they are spring values
        for date in result.index:
            assert (date.month == 3 and date.day >= 14) or (
                date.month == 4 and date.day < 15
            )


def test_gg(biweekly_timeseries: pd.Series) -> None:
    """Test gg function."""
    # Test default parameters
    result = ps.stats.gg(biweekly_timeseries, min_n_years=1, min_n_meas=1)
    assert isinstance(result, float)

    # Test different outputs
    result_yearly = ps.stats.gg(
        biweekly_timeseries, output="yearly", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_yearly, pd.Series)

    result_semi = ps.stats.gg(
        biweekly_timeseries, output="semimonthly", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_semi, pd.Series)

    # Verify average calculation
    mean_value = biweekly_timeseries.mean()
    # GG should be somewhat close to the mean of all values
    assert abs(result - mean_value) < 1.0


def test_ghg_outputs(biweekly_timeseries: pd.Series) -> None:
    """Test ghg function with various output formats."""
    # Test default parameters (mean output)
    result_mean = ps.stats.ghg(biweekly_timeseries, min_n_years=1, min_n_meas=1)
    assert isinstance(result_mean, float)

    # Test yearly output
    result_yearly = ps.stats.ghg(
        biweekly_timeseries, output="yearly", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_yearly, pd.Series)

    # Test g3 output (selected data points used for calculation)
    result_g3 = ps.stats.ghg(
        biweekly_timeseries, output="g3", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_g3, pd.Series)

    # Test semimonthly output
    result_semi = ps.stats.ghg(
        biweekly_timeseries, output="semimonthly", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_semi, pd.Series)

    # Verify relationship between outputs
    if not result_yearly.empty and not np.isnan(result_mean):
        assert abs(result_yearly.mean() - result_mean) < 1e-10


def test_glg_outputs(biweekly_timeseries: pd.Series) -> None:
    """Test glg function with various output formats."""
    # Test default parameters (mean output)
    result_mean = ps.stats.glg(biweekly_timeseries, min_n_years=1, min_n_meas=1)
    assert isinstance(result_mean, float)

    # Test yearly output
    result_yearly = ps.stats.glg(
        biweekly_timeseries, output="yearly", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_yearly, pd.Series)

    # Test g3 output (selected data points used for calculation)
    result_g3 = ps.stats.glg(
        biweekly_timeseries, output="g3", min_n_years=1, min_n_meas=1
    )
    assert isinstance(result_g3, pd.Series)

    # Verify relationship between outputs
    if not result_yearly.empty and not np.isnan(result_mean):
        assert abs(result_yearly.mean() - result_mean) < 1e-10


def test_gxg_min_requirements() -> None:
    """Test minimum requirements parameters for gxg functions."""
    # Create a short series that doesn't meet min_n_years requirement
    short_index = pd.date_range("2020-01-01", "2020-12-31", freq="14D")
    short_series = pd.Series(np.random.randn(len(short_index)), index=short_index)

    # Test with default requirements (should return NaN)
    assert np.isnan(ps.stats.ghg(short_series))

    # Test with relaxed requirements
    result = ps.stats.ghg(short_series, min_n_years=1, min_n_meas=4)
    assert isinstance(result, float) and not np.isnan(result)

    # Create a sparse series that doesn't meet min_n_meas requirement
    sparse_index = pd.date_range("2018-01-01", "2021-12-31", freq="90D")
    sparse_series = pd.Series(np.random.randn(len(sparse_index)), index=sparse_index)

    # Test with default requirements (should return NaN)
    assert np.isnan(ps.stats.glg(sparse_series))

    # Test with relaxed requirements
    result = ps.stats.glg(sparse_series, min_n_meas=1, min_n_years=3)
    assert isinstance(result, float) and not np.isnan(result)


def test_fill_methods(sample_timeseries: pd.Series) -> None:
    """Test different fill methods for gxg functions."""
    # Create series with strategic gaps
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="14D")
    values = np.sin(np.arange(len(dates)) * 0.2) * 10 + 100
    series = pd.Series(values, index=dates)

    # Add gaps
    mask = (series.index.month % 3 == 0) & (series.index.day < 15)
    series[mask] = np.nan

    # Test with different fill methods
    methods = ["ffill", "bfill", "nearest", "linear", None]
    results = {}

    for method in methods:
        results[method] = ps.stats.ghg(
            series, fill_method=method, limit=10, min_n_years=1, min_n_meas=4
        )

    # Verify that different methods give different results
    methods_with_values = [m for m in methods if not np.isnan(results[m])]
    if len(methods_with_values) > 1:
        values_set = set([round(results[m], 5) for m in methods_with_values])
        # At least some methods should produce different results
        assert len(values_set) > 1


class TestGXG(object):
    def test_ghg(self) -> None:
        idx = pd.to_datetime(["20160114", "20160115", "20160128", "20160214"])
        s = pd.Series([10.0, 3.0, 30.0, 20.0], index=idx)
        v = ps.stats.ghg(s, min_n_meas=1, min_n_years=1)
        assert v == 30.0

    def test_ghg_ffill(self) -> None:
        idx = pd.to_datetime(["20160101", "20160115", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="ffill", limit=15, min_n_meas=1, min_n_years=1)
        assert v == 0.0

    def test_ghg_bfill(self) -> None:
        idx = pd.to_datetime(["20160101", "20160115", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="bfill", limit=15, min_n_meas=1, min_n_years=1)
        # TODO is this correct?
        assert v == 10.0

    def test_ghg_linear(self) -> None:
        idx = pd.to_datetime(["20160101", "20160110", "20160120", "20160130"])
        s = pd.Series([0.0, 0.0, 10.0, 10.0], index=idx)
        v = ps.stats.ghg(s, fill_method="linear", min_n_meas=1, min_n_years=1, limit=8)
        # TODO is this correct?
        assert v == 10.0

    def test_ghg_len_yearly(self) -> None:
        idx = pd.date_range("20000101", "20550101", freq="d")
        s = pd.Series(np.ones(len(idx)), index=idx)
        v = ps.stats.ghg(s, output="yearly")
        assert v.notna().sum() == 55

    def test_glg(self) -> None:
        idx = pd.date_range("20000101", "20550101", freq="d")
        s = pd.Series(
            [x.month + x.day for x in idx],
            index=idx,
        )
        v = ps.stats.glg(s, year_offset="YE")
        assert v == 16.0

    def test_glg_fill_limit(self) -> None:
        idx = pd.to_datetime(["20170115", "20170130", "20200101"])
        s = pd.Series(np.ones(len(idx)), index=idx)
        v = ps.stats.glg(
            s,
            fill_method="linear",
            limit=15,
            output="yearly",
            year_offset="YE",
            min_n_meas=1,
        )
        assert v.notna().sum() == 2

    def test_gvg(self) -> None:
        idx = pd.to_datetime(["20170314", "20170328", "20170414", "20170428"])
        s = pd.Series([1.0, 2.0, 3.0, 4], index=idx)
        v = ps.stats.gvg(
            s, fill_method="linear", output="mean", min_n_meas=1, min_n_years=1
        )
        assert v == 2.0

    def test_gvg_nan(self) -> None:
        idx = pd.to_datetime(["20170228", "20170428", "20170429"])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        v = ps.stats.gvg(
            s, fill_method=None, output="mean", min_n_meas=1, min_n_years=1
        )
        assert np.isnan(v)


class TestQGXG(object):
    def test_q_ghg(self) -> None:
        n = 101
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_ghg(s, q=0.94)
        assert v == 94.0

    def test_q_glg(self) -> None:
        n = 101
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_glg(s, q=0.06)
        assert v == 6.0

    def test_q_ghg_nan(self) -> None:
        idx = pd.date_range("20160101", freq="d", periods=4)
        s = pd.Series([1, np.nan, 3, np.nan], index=idx)
        v = ps.stats.q_ghg(s, q=0.5)
        assert v == 2.0

    def test_q_gvg(self) -> None:
        idx = pd.to_datetime(["20160320", "20160401", "20160420"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s)
        assert v == 2.5

    def test_q_gvg_nan(self) -> None:
        idx = pd.to_datetime(["20160820", "20160901", "20161120"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s)
        assert np.isnan(v)

    def test_q_glg_tmin(self) -> None:
        tmin = "20160301"
        idx = pd.date_range("20160101", "20160331", freq="d")
        s = pd.Series(np.arange(len(idx)), index=idx)
        v = ps.stats.q_glg(s, q=0.06, tmin=tmin)
        assert v == 61.8

    def test_q_ghg_tmax(self) -> None:
        n = 101
        tmax = "20160301"
        idx = pd.date_range("20160101", freq="d", periods=n)
        s = pd.Series(np.arange(n), index=idx)
        v = ps.stats.q_ghg(s, q=0.94, tmax=tmax)
        assert v == 56.4

    def test_q_gvg_tmin_tmax(self) -> None:
        tmin = "20170301"
        tmax = "20170401"
        idx = pd.to_datetime(["20160401", "20170401", "20180401"])
        s = pd.Series([0, 5, 10], index=idx)
        v = ps.stats.q_gvg(s, tmin=tmin, tmax=tmax)
        assert v == 5

    def test_q_gxg_series(self) -> None:
        """Test q_gxg functions against reference values from Menyanthes."""
        s = pd.read_csv(
            "tests/data/hseries_gxg.csv",
            index_col=0,
            header=0,
            parse_dates=True,
            dayfirst=True,
        ).squeeze("columns")

        # Calculate GXG values
        ghg = ps.stats.q_ghg(s)
        glg = ps.stats.q_glg(s)
        gvg = ps.stats.q_gvg(s)

        # Reference values from Menyanthes
        ref_ghg = -3.23
        ref_glg = -3.82
        ref_gvg = -3.43

        # Assert that calculated values are close to reference values
        # Using tolerance of 0.1m which is reasonable for groundwater levels
        assert abs(ghg - ref_ghg) < 0.1, f"GHG expected {ref_ghg}, got {ghg:.2f}"
        assert abs(glg - ref_glg) < 0.1, f"GLG expected {ref_glg}, got {glg:.2f}"
        assert abs(gvg - ref_gvg) < 0.1, f"GVG expected {ref_gvg}, got {gvg:.2f}"
