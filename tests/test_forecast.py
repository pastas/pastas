import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, DatetimeIndex, MultiIndex, Series

from pastas import Model
from pastas.forecast import (
    _check_forecast_data,
    forecast,
    get_overall_mean_and_variance,
)


class TestCheckForecastData:
    def test_empty_dict(self) -> None:
        """Test that an empty dictionary raises a ValueError."""
        with pytest.raises(
            ValueError, match="Forecasts must be a non-empty dictionary"
        ):
            _check_forecast_data({})

    def test_not_a_dict(self) -> None:
        """Test that a non-dictionary input raises a ValueError."""
        with pytest.raises(
            ValueError, match="Forecasts must be a non-empty dictionary"
        ):
            _check_forecast_data([])

    def test_no_valid_forecast_data(self) -> None:
        """Test that no valid forecast data raises a ValueError."""
        forecasts: dict[str, list[DataFrame]] = {"sm1": []}
        with pytest.raises(ValueError, match="Forecast data for stressmodel"):
            _check_forecast_data(forecasts)

    def test_empty_dataframe(self) -> None:
        """Test that empty DataFrames are handled correctly."""
        empty_df = pd.DataFrame()
        forecasts: dict[str, dict[str, DataFrame]] = {"sm1": {"prec": empty_df}}
        with pytest.raises(
            ValueError, match="No valid forecast data found in any of the stressmodels"
        ):
            _check_forecast_data(forecasts)

    def test_mismatched_column_counts(self) -> None:
        """Test that DataFrames with different column counts raise a ValueError."""
        index: DatetimeIndex = pd.date_range("2023-01-01", periods=10, freq="D")
        df1: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        df2: DataFrame = pd.DataFrame(np.random.rand(10, 4), index=index)
        forecasts: dict[str, dict[str, DataFrame]] = {"sm1": {"prec": df1, "evap": df2}}
        with pytest.raises(
            ValueError, match="number of ensemble members is not the same"
        ):
            _check_forecast_data(forecasts)

    def test_mismatched_indices(self) -> None:
        """Test that DataFrames with different indices raise a ValueError."""
        index1: DatetimeIndex = pd.date_range("2023-01-01", periods=10, freq="D")
        index2: DatetimeIndex = pd.date_range("2023-01-02", periods=10, freq="D")
        df1: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index1)
        df2: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index2)
        forecasts: dict[str, dict[str, DataFrame]] = {"sm1": {"prec": df1, "evap": df2}}
        with pytest.raises(
            ValueError, match="time index of the forecasts is not the same"
        ):
            _check_forecast_data(forecasts)

    def test_valid_forecast_data(self) -> None:
        """Test that valid forecast data returns the correct values."""
        index: DatetimeIndex = pd.date_range("2023-01-01", periods=10, freq="D")
        df1: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        df2: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        forecasts: dict[str, dict[str, DataFrame]] = {"sm1": {"prec": df1, "evap": df2}}
        n, tmin, tmax, result_index = _check_forecast_data(forecasts)
        assert n == 3
        assert tmin == index[0]
        assert tmax == index[-1]
        assert (result_index == index).all()


class TestGetOverallMeanAndVariance:
    index: DatetimeIndex
    df: DataFrame

    def setup_method(self) -> None:
        """Set up test data for get_overall_mean_and_variance."""
        self.index = pd.date_range("2023-01-01", periods=5, freq="D")
        columns: MultiIndex = pd.MultiIndex.from_product(
            [range(2), range(3), ["mean", "var"]],
            names=["ensemble_member", "param_member", "forecast"],
        )

        # Create data with known mean and variance patterns
        # For ensemble_member=0, param_member=0, mean values are 1.0
        # For ensemble_member=0, param_member=1, mean values are 2.0
        # For ensemble_member=0, param_member=2, mean values are 3.0
        # For ensemble_member=1, param_member=0, mean values are 4.0
        # For ensemble_member=1, param_member=1, mean values are 5.0
        # For ensemble_member=1, param_member=2, mean values are 6.0
        # All var values are 0.5

        data: np.ndarray = np.zeros((5, 12))
        for i in range(0, 12, 2):
            data[:, i] = i / 2 + 1  # mean values
            data[:, i + 1] = 0.5  # variance values

        self.df = pd.DataFrame(data=data, index=self.index, columns=columns)

    def test_overall_mean_calculation(self) -> None:
        """Test that the overall mean is calculated correctly."""
        mean, _ = get_overall_mean_and_variance(self.df)

        # Expected mean is average of all mean values (1+2+3+4+5+6)/6 = 3.5
        assert isinstance(mean, Series)
        assert len(mean) == len(self.index)
        assert (mean == 3.5).all()

    def test_overall_variance_calculation(self) -> None:
        """Test that the overall variance is calculated correctly."""
        _, var = get_overall_mean_and_variance(self.df)

        # Variance of means: ((1-3.5)² + (2-3.5)² + ... + (6-3.5)²)/5 = 3.5 # ddof=1
        # Mean of variances: 0.5
        # Expected total variance = 3.5 + 0.5 = 4
        assert isinstance(var, Series)
        assert len(var) == len(self.index)
        assert (var == 4.0).all()


@pytest.fixture
def forecast_data() -> dict[str, list[DataFrame]]:
    """Create forecast data for testing."""
    index: DatetimeIndex = pd.date_range("2015-11-30", periods=5, freq="D")
    df1: DataFrame = pd.DataFrame(np.ones((5, 2)), index=index)
    df2: DataFrame = pd.DataFrame(np.ones((5, 2)) * 2, index=index)

    return {"rch": {"prec": df1, "evap": df2}}


class TestForecast:
    def test_forecast_no_noisemodel_with_post_processing(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test forecast with post-processing but no noise model."""
        # Remove noise model
        ml_noisemodel.noisemodel = None

        with pytest.raises(ValueError, match="No noise model present"):
            forecast(ml_noisemodel, forecast_data, post_process=True)

    def test_forecast_empty_params(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test forecast with empty params list."""
        with pytest.raises(ValueError, match="Empty parameter list provided"):
            forecast(ml_noisemodel, forecast_data, p=[])

    def test_forecast_valid_input_post_process(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test forecast with valid input and post-processing enabled."""
        # Ensure noise model is present
        print(ml_noisemodel.oseries)
        result = forecast(ml_noisemodel, forecast_data, post_process=True)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_forecast_valid_input_no_post_process(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test forecast with valid input and post-processing disabled."""
        result = forecast(ml_noisemodel, forecast_data, post_process=False)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_forecast_missing_key(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test forecast with missing required key in forecast_data."""
        # Remove the key expected by the model (simulate missing stressmodel)
        bad_data = {"wrong_key": forecast_data["rch"]}
        with pytest.raises(Exception):
            forecast(ml_noisemodel, bad_data)

    def test_forecast_output_shape(
        self, ml_noisemodel: Model, forecast_data: dict[str, dict[str, DataFrame]]
    ) -> None:
        """Test that forecast output has expected shape."""
        result = forecast(ml_noisemodel, forecast_data)
        # Should have same index as input and at least one column
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 5
        assert result.shape[1] >= 1
