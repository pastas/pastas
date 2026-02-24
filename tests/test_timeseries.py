import numpy as np
import pandas as pd
import pytest

from pastas.timeseries import TimeSeries, validate_oseries, validate_stress


@pytest.fixture
def daily_series() -> pd.Series:
    """Create a daily time series fixture."""
    index = pd.date_range("2000-01-01", periods=100, freq="D")
    data = np.random.rand(100)
    series = pd.Series(data=data, index=index, name="test_series")
    return series


@pytest.fixture
def hourly_series() -> pd.Series:
    """Create an hourly time series fixture."""
    index = pd.date_range("2000-01-01", periods=100, freq="h")
    data = np.random.rand(100)
    series = pd.Series(data=data, index=index, name="hourly_series")
    return series


@pytest.fixture
def series_with_nans() -> pd.Series:
    """Create a series with NaN values."""
    index = pd.date_range("2000-01-01", periods=100, freq="D")
    data = np.random.rand(100)
    data[10:15] = np.nan  # Add some NaN values
    series = pd.Series(data=data, index=index, name="series_with_nans")
    return series


def test_timeseries_init(daily_series: pd.Series) -> None:
    """Test TimeSeries initialization."""
    ts = TimeSeries(daily_series)
    assert ts.name == "test_series"
    assert ts.freq_original == "1D"
    assert ts.settings["freq"] == "1D"
    assert ts._series is not None
    assert ts._series_original is not None


def test_timeseries_init_with_settings(daily_series: pd.Series) -> None:
    """Test TimeSeries initialization with custom settings."""
    settings = {
        "fill_nan": "interpolate",
        "fill_before": "mean",
        "fill_after": "mean",
    }
    ts = TimeSeries(daily_series, settings=settings)
    assert ts.settings["fill_nan"] == "interpolate"
    assert ts.settings["fill_before"] == "mean"
    assert ts.settings["fill_after"] == "mean"


def test_timeseries_init_with_predefined_settings(daily_series: pd.Series) -> None:
    """Test TimeSeries initialization with predefined settings string."""
    ts = TimeSeries(daily_series, settings="oseries")
    # Assert that predefined settings were applied
    assert ts.settings != {}


def test_timeseries_init_with_name(daily_series: pd.Series) -> None:
    """Test TimeSeries initialization with custom name."""
    ts = TimeSeries(daily_series, name="custom_name")
    assert ts.name == "custom_name"


def test_timeseries_init_with_metadata(daily_series: pd.Series) -> None:
    """Test TimeSeries initialization with metadata."""
    metadata = {"location": "test", "units": "m"}
    ts = TimeSeries(daily_series, metadata=metadata)
    assert ts.metadata["location"] == "test"
    assert ts.metadata["units"] == "m"


def test_timeseries_update_series(daily_series: pd.Series) -> None:
    """Test updating series settings."""
    ts = TimeSeries(daily_series, settings="prec")
    # Update frequency
    ts.update_series(freq="7D")
    assert ts.settings["freq"] == "7D"
    # The series should be resampled
    assert len(ts.series) < len(daily_series)


def test_timeseries_update_tmin_tmax(daily_series: pd.Series) -> None:
    """Test updating tmin and tmax."""
    ts = TimeSeries(daily_series)
    new_tmin = "2000-01-15"
    new_tmax = "2000-03-15"
    ts.update_series(tmin=new_tmin, tmax=new_tmax)
    assert ts.settings["tmin"] == pd.Timestamp(new_tmin)
    assert ts.settings["tmax"] == pd.Timestamp(new_tmax)
    assert ts.series.index.min() >= pd.Timestamp(new_tmin)
    assert ts.series.index.max() <= pd.Timestamp(new_tmax)


def test_fill_nan_methods(series_with_nans: pd.Series) -> None:
    """Test different fill_nan methods."""
    # Test "interpolate"
    ts_interpolate = TimeSeries(series_with_nans, settings={"fill_nan": "interpolate"})
    assert not ts_interpolate.series.isna().any()

    # Test "mean"
    ts_mean = TimeSeries(series_with_nans, settings={"fill_nan": "mean"})
    assert not ts_mean.series.isna().any()

    # Test float value
    ts_float = TimeSeries(series_with_nans, settings={"fill_nan": 0.0})
    assert not ts_float.series.isna().any()
    assert (ts_float.series[10:15] == 0.0).all()


def test_sample_up_methods(daily_series: pd.Series) -> None:
    """Test different sample_up methods."""
    ts = TimeSeries(daily_series)

    # Test "interpolate"
    ts.update_series(freq="12h", sample_up="interpolate")
    assert len(ts.series) > len(daily_series)

    # Test "bfill"
    ts.update_series(freq="12h", sample_up="bfill")
    assert len(ts.series) > len(daily_series)

    # Test "ffill"
    ts.update_series(freq="12h", sample_up="ffill")
    assert len(ts.series) > len(daily_series)

    # Test "mean"
    ts.update_series(freq="12h", sample_up="mean")
    assert len(ts.series) > len(daily_series)

    # Test float value
    ts.update_series(freq="12h", sample_up=0.0)
    assert len(ts.series) > len(daily_series)


def test_sample_down_methods(hourly_series: pd.Series) -> None:
    """Test different sample_down methods."""
    ts = TimeSeries(hourly_series)

    # Test "mean"
    ts.update_series(freq="D", sample_down="mean")
    assert len(ts.series) < len(hourly_series)

    # Test "sum"
    ts.update_series(freq="D", sample_down="sum")
    assert len(ts.series) < len(hourly_series)

    # Test "min"
    ts.update_series(freq="D", sample_down="min")
    assert len(ts.series) < len(hourly_series)

    # Test "max"
    ts.update_series(freq="D", sample_down="max")
    assert len(ts.series) < len(hourly_series)


def test_fill_before_methods(daily_series: pd.Series) -> None:
    """Test different fill_before methods."""
    # Create a time series that starts later
    ts = TimeSeries(daily_series)

    # Test "mean"
    ts.update_series(tmin="1999-12-01", fill_before="mean")
    assert ts.series.index.min() < daily_series.index.min()

    # Test "bfill"
    ts.update_series(tmin="1999-12-01", fill_before="bfill")
    assert ts.series.index.min() < daily_series.index.min()

    # Test float value
    ts.update_series(tmin="1999-12-01", fill_before=0.0)
    assert ts.series.index.min() < daily_series.index.min()


def test_fill_after_methods(daily_series: pd.Series) -> None:
    """Test different fill_after methods."""
    ts = TimeSeries(daily_series)

    # Test "mean"
    ts.update_series(tmax="2000-05-01", fill_after="mean")
    assert ts.series.index.max() > daily_series.index.max()

    # Test "ffill"
    ts.update_series(tmax="2000-05-01", fill_after="ffill")
    assert ts.series.index.max() > daily_series.index.max()

    # Test float value
    ts.update_series(tmax="2000-05-01", fill_after=0.0)
    assert ts.series.index.max() > daily_series.index.max()


def test_to_dict(daily_series: pd.Series) -> None:
    """Test to_dict method."""
    ts = TimeSeries(daily_series)

    # With series=True
    result = ts.to_dict(series=True)
    assert "series" in result
    assert "name" in result
    assert "settings" in result
    assert "metadata" in result

    # With series=False
    result = ts.to_dict(series=False)
    assert "series" not in result


def test_validate_stress(daily_series: pd.Series) -> None:
    """Test validate_stress function."""
    # Valid series should pass
    assert validate_stress(daily_series) is True

    # Test with non-equidistant series
    irregular_index = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-05"])
    irregular_series = pd.Series([1.0, 2.0, 3.0], index=irregular_index)
    with pytest.raises(ValueError):
        validate_stress(irregular_series)


def test_validate_oseries(daily_series: pd.Series) -> None:
    """Test validate_oseries function."""
    # Valid series should pass
    assert validate_oseries(daily_series) is True

    # Test with invalid index
    series_with_invalid_index = daily_series.copy()
    series_with_invalid_index.index = range(len(daily_series))
    with pytest.raises(ValueError):
        validate_oseries(series_with_invalid_index)

    # Test with duplicate indices
    dup_index = daily_series.index.append(pd.DatetimeIndex([daily_series.index[0]]))
    dup_data = np.append(daily_series.values, [daily_series.values[0]])
    dup_series = pd.Series(dup_data, index=dup_index)
    with pytest.raises(ValueError):
        validate_oseries(dup_series)


def test_series_original_setter(
    daily_series: pd.Series, hourly_series: pd.Series
) -> None:
    """Test setting series_original property."""
    ts = TimeSeries(daily_series)
    assert ts.freq_original == "1D"

    # Update original series
    ts.series_original = hourly_series
    assert ts.freq_original == "h"
    assert ts._series_original.equals(hourly_series)


def test_series_setter_raises_error(daily_series: pd.Series) -> None:
    """Test that setting series directly raises an error."""
    ts = TimeSeries(daily_series)
    with pytest.raises(AttributeError):
        ts.series = daily_series
