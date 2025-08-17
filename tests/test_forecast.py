from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, DatetimeIndex, MultiIndex, Series, Timestamp

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
        forecasts: Dict[str, List[DataFrame]] = {"sm1": []}
        with pytest.raises(ValueError, match="No valid forecast data found"):
            _check_forecast_data(forecasts)

    def test_empty_dataframe(self) -> None:
        """Test that empty DataFrames are handled correctly."""
        empty_df = pd.DataFrame()
        forecasts: Dict[str, List[DataFrame]] = {"sm1": [empty_df]}
        with pytest.raises(ValueError, match="No valid forecast data found"):
            _check_forecast_data(forecasts)

    def test_mismatched_column_counts(self) -> None:
        """Test that DataFrames with different column counts raise a ValueError."""
        index: DatetimeIndex = pd.date_range("2023-01-01", periods=10, freq="D")
        df1: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        df2: DataFrame = pd.DataFrame(np.random.rand(10, 4), index=index)
        forecasts: Dict[str, List[DataFrame]] = {"sm1": [df1, df2]}
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
        forecasts: Dict[str, List[DataFrame]] = {"sm1": [df1, df2]}
        with pytest.raises(
            ValueError, match="time index of the forecasts is not the same"
        ):
            _check_forecast_data(forecasts)

    def test_valid_forecast_data(self) -> None:
        """Test that valid forecast data returns the correct values."""
        index: DatetimeIndex = pd.date_range("2023-01-01", periods=10, freq="D")
        df1: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        df2: DataFrame = pd.DataFrame(np.random.rand(10, 3), index=index)
        forecasts: Dict[str, List[DataFrame]] = {"sm1": [df1, df2]}
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
def mock_model() -> MagicMock:
    """Create a mock Pastas model for testing forecast."""
    model = MagicMock()
    model.copy.return_value = model
    model.settings = {"freq_obs": pd.Timedelta("1D")}
    model.noisemodel = MagicMock()

    # Mock stressmodel with properly configured series
    sm1 = MagicMock()
    sm1.stress = MagicMock()
    sm1.stress.__getitem__.return_value = MagicMock()
    sm1.stress.__getitem__().series_original = pd.Series(
        np.ones(5), index=pd.date_range("2023-01-01", "2023-01-05", freq="D")
    )
    model.stressmodels = {"sm1": sm1}

    # Mock the solver
    solver = MagicMock()
    solver.get_parameter_sample.return_value = np.array(
        [[1.0, 2.0, 50.0], [3.0, 4.0, 60.0]]
    )
    model.solver = solver

    # Mock simulate method
    def simulate(*args: Any, **kwargs: Any) -> Series:
        tmin: Timestamp = kwargs.get("tmin", pd.Timestamp("2023-01-01"))
        tmax: Timestamp = kwargs.get("tmax", pd.Timestamp("2023-01-05"))
        index: DatetimeIndex = pd.date_range(tmin, tmax, freq="D")
        return pd.Series(np.ones(len(index)), index=index)

    model.simulate = simulate

    # Mock residuals method
    def residuals(*args: Any, **kwargs: Any) -> Series:
        index: DatetimeIndex = pd.date_range("2022-12-20", "2022-12-31", freq="D")
        return pd.Series(np.ones(len(index)), index=index)

    model.residuals = residuals

    # Mock noise method
    def noise(*args: Any, **kwargs: Any) -> Series:
        index: DatetimeIndex = pd.date_range("2022-12-20", "2022-12-31", freq="D")
        return pd.Series(np.ones(len(index)) * 0.25, index=index)

    model.noise = noise

    return model


@pytest.fixture
def forecast_data() -> Dict[str, List[DataFrame]]:
    """Create forecast data for testing."""
    index: DatetimeIndex = pd.date_range("2023-01-01", periods=5, freq="D")
    df1: DataFrame = pd.DataFrame(np.ones((5, 2)), index=index)
    df2: DataFrame = pd.DataFrame(np.ones((5, 2)) * 2, index=index)

    return {"sm1": [df1, df2]}


class TestForecast:
    def test_forecast_without_params(
        self, mock_model: MagicMock, forecast_data: Dict[str, List[DataFrame]]
    ) -> None:
        """Test forecast without providing parameters."""
        with patch("pastas.forecast._check_forecast_data") as mock_check:
            mock_check.return_value = (
                2,
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-05"),
                pd.date_range("2023-01-01", "2023-01-05", freq="D"),
            )

            # Set up mock for get_correction
            mock_model.noisemodel.get_correction.return_value = pd.Series(
                np.zeros(5), index=pd.date_range("2023-01-01", "2023-01-05", freq="D")
            )

            # Run forecast
            result: DataFrame = forecast(mock_model, forecast_data, nparam=2)

            # Check results
            assert isinstance(result, DataFrame)
            assert isinstance(result.columns, MultiIndex)
            assert result.columns.names == [
                "ensemble_member",
                "param_member",
                "forecast",
            ]
            assert set(result.columns.get_level_values(2)) == {"mean", "var"}
            assert result.shape == (
                5,
                8,
            )  # 5 days, 2 ensemble members × 2 params × 2 (mean, var)

    def test_forecast_with_params(
        self, mock_model: MagicMock, forecast_data: Dict[str, List[DataFrame]]
    ) -> None:
        """Test forecast with provided parameters."""
        with patch("pastas.forecast._check_forecast_data") as mock_check:
            mock_check.return_value = (
                2,
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-05"),
                pd.date_range("2023-01-01", "2023-01-05", freq="D"),
            )

            # Mock get_correction
            mock_model.noisemodel.get_correction.return_value = pd.Series(
                np.zeros(5), index=pd.date_range("2023-01-01", "2023-01-05", freq="D")
            )

            params: List[List[float]] = [[1.0, 2.0, 50.0], [3.0, 4.0, 60.0]]
            result: DataFrame = forecast(mock_model, forecast_data, params=params)

            assert isinstance(result, DataFrame)
            assert result.shape == (
                5,
                8,
            )  # 5 days, 2 ensemble members × 2 params × 2 (mean, var)

    def test_forecast_with_post_processing(
        self, mock_model: MagicMock, forecast_data: Dict[str, List[DataFrame]]
    ) -> None:
        """Test forecast with post-processing."""
        with patch("pastas.forecast._check_forecast_data") as mock_check:
            mock_check.return_value = (
                2,
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-01-05"),
                pd.date_range("2023-01-01", "2023-01-05", freq="D"),
            )

            # Mock get_correction to return series with known values
            mock_model.noisemodel.get_correction.return_value = pd.Series(
                np.ones(5) * 0.5,
                index=pd.date_range("2023-01-01", "2023-01-05", freq="D"),
            )

            params: List[List[float]] = [[1.0, 2.0, 50.0], [3.0, 4.0, 60.0]]
            result: DataFrame = forecast(
                mock_model, forecast_data, params=params, post_process=True
            )

            assert isinstance(result, DataFrame)
            assert result.shape == (
                5,
                8,
            )  # 5 days, 2 ensemble members × 2 params × 2 (mean, var)

            # Check that corrections were applied (values should be 1.0 + 0.5 = 1.5)
            means: DataFrame = result.loc[:, (slice(None), slice(None), "mean")]
            assert np.allclose(means.iloc[0, 0], 1.5)

    def test_forecast_no_noisemodel_with_post_processing(
        self, mock_model: MagicMock, forecast_data: Dict[str, List[DataFrame]]
    ) -> None:
        """Test forecast with post-processing but no noise model."""
        # Remove noise model
        mock_model.noisemodel = None

        with pytest.raises(ValueError, match="No noisemodel is present"):
            forecast(mock_model, forecast_data, post_process=True)

    def test_forecast_empty_params(
        self, mock_model: MagicMock, forecast_data: Dict[str, List[DataFrame]]
    ) -> None:
        """Test forecast with empty params list."""
        with pytest.raises(ValueError, match="Empty parameter list provided"):
            forecast(mock_model, forecast_data, params=[])
