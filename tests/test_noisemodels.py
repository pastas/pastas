"""Tests for the noise models in pastas."""

import numpy as np
import pandas as pd
import pytest
from pandas import Series

from pastas.noisemodels import ArmaNoiseModel, ArNoiseModel


@pytest.fixture
def residual_series() -> Series:
    """Create a residual series for testing."""
    # Create daily data for 100 days
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # Create a residual series with some pattern
    values = np.sin(np.linspace(0, 2 * np.pi * 3, 100)) + np.random.normal(0, 0.1, 100)
    return Series(values, index=dates, name="Residuals")


@pytest.fixture
def irregular_residual_series() -> Series:
    """Create an irregular residual series for testing."""
    # Create 100 dates with irregular spacing
    base_dates = pd.date_range(start="2020-01-01", periods=150, freq="D")
    # Select random 100 dates (sorted)
    selected_indices = np.sort(np.random.choice(150, 100, replace=False))
    dates = base_dates[selected_indices]

    # Create residuals with some pattern
    values = np.sin(np.linspace(0, 2 * np.pi * 3, 100)) + np.random.normal(0, 0.1, 100)
    return Series(values, index=dates, name="Residuals")


class TestArNoiseModel:
    """Test the AR noise model."""

    def test_init(self) -> None:
        """Test initialization of AR noise model."""
        model = ArNoiseModel()
        assert model._name == "ArNoiseModel"
        assert model.nparam == 1
        assert model.norm is True
        assert "noise_alpha" in model.parameters.index

        # Test initialization with norm=False
        model = ArNoiseModel(norm=False)
        assert not model.norm

    def test_set_init_parameters(self) -> None:
        """Test setting initial parameters with and without oseries."""
        # Without oseries
        model = ArNoiseModel()
        model.set_init_parameters()
        assert model.parameters.loc["noise_alpha", "initial"] == 14.0

        # With oseries
        oseries = pd.Series(
            index=pd.date_range(start="2020-01-01", periods=10, freq="3D")
        )
        model = ArNoiseModel()
        model.set_init_parameters(oseries)
        assert model.parameters.loc["noise_alpha", "initial"] == 3.0  # 3D frequency

    def test_simulate(self, residual_series: Series) -> None:
        """Test noise simulation."""
        model = ArNoiseModel()
        alpha = 10.0  # Set alpha parameter

        # Simulate noise
        noise = model.simulate(residual_series, [alpha])

        # Check properties
        assert len(noise) == len(residual_series)
        assert isinstance(noise, Series)
        assert noise.name == "Noise"

        # First value should equal first residual
        assert noise.iloc[0] == residual_series.iloc[0]

        # Calculate expected noise for second value
        dt = (residual_series.index[1] - residual_series.index[0]).days
        expected_noise = (
            residual_series.iloc[1] - np.exp(-dt / alpha) * residual_series.iloc[0]
        )
        assert noise.iloc[1] == pytest.approx(expected_noise)

    def test_weights(self, residual_series: Series) -> None:
        """Test weights calculation."""
        model = ArNoiseModel()
        alpha = 10.0

        # Get weights
        weights = model.weights(residual_series, [alpha])

        # Check basic properties
        assert len(weights) == len(residual_series)
        assert isinstance(weights, Series)
        assert weights.name == "noise_weights"

        # Test with normalization off
        model.norm = False
        weights_no_norm = model.weights(residual_series, [alpha])

        # Should be different from normalized weights
        assert not np.allclose(weights.values, weights_no_norm.values)

    def test_get_correction(self, residual_series: Series) -> None:
        """Test forecast correction."""
        model = ArNoiseModel()
        alpha = 10.0

        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=residual_series.index[-1] + pd.Timedelta(days=1), periods=10, freq="D"
        )

        # Get correction
        correction = model.get_correction(residual_series, [alpha], forecast_dates)

        # Check properties
        assert len(correction) == len(forecast_dates)
        assert correction.name == "correction"

        # Check calculation
        last_residual = residual_series.iloc[-1]
        dt_days = 1  # First forecast is 1 day away
        expected_first_correction = np.exp(-dt_days / alpha) * last_residual
        assert correction.iloc[0] == pytest.approx(expected_first_correction)

        # Correction should decay with time
        assert abs(correction.iloc[-1]) < abs(correction.iloc[0])

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        model = ArNoiseModel(norm=False)
        data = model.to_dict()

        assert data["class"] == "ArNoiseModel"
        assert data["norm"] is False


class TestArmaNoiseModel:
    """Test the ARMA noise model."""

    def test_init(self) -> None:
        """Test initialization of ARMA noise model."""
        model = ArmaNoiseModel()
        assert model._name == "ArmaNoiseModel"
        assert model.nparam == 2
        assert "noise_alpha" in model.parameters.index
        assert "noise_beta" in model.parameters.index

    def test_set_init_parameters(self) -> None:
        """Test setting initial parameters with and without oseries."""
        # Without oseries
        model = ArmaNoiseModel()
        model.set_init_parameters()
        assert model.parameters.loc["noise_alpha", "initial"] == 14.0
        assert model.parameters.loc["noise_beta", "initial"] == 1.0

        # With oseries
        oseries = pd.Series(
            index=pd.date_range(start="2020-01-01", periods=10, freq="7D")
        )
        model = ArmaNoiseModel()
        model.set_init_parameters(oseries)
        assert model.parameters.loc["noise_alpha", "initial"] == 7.0  # 7D frequency
        assert model.parameters.loc["noise_beta", "initial"] == 1.0

    def test_simulate(self, residual_series: Series) -> None:
        """Test noise simulation."""
        model = ArmaNoiseModel()
        params = [10.0, 5.0]  # alpha, beta

        # Simulate noise
        noise = model.simulate(residual_series, params)

        # Check properties
        assert len(noise) == len(residual_series)
        assert isinstance(noise, Series)
        assert noise.name == "Noise"

        # First value should equal first residual
        assert noise.iloc[0] == residual_series.iloc[0]

    def test_calculate_noise_edge_cases(self, residual_series: Series) -> None:
        """Test noise calculation with edge case parameters."""
        model = ArmaNoiseModel()

        # Test with beta = 0 (should use a small value instead)
        params = [10.0, 0.0]  # alpha, beta
        noise_beta_zero = model.simulate(residual_series, params)
        assert np.isfinite(noise_beta_zero).all()

        # Test with negative beta (should handle sign correctly)
        params = [10.0, -5.0]  # alpha, beta
        noise_beta_neg = model.simulate(residual_series, params)
        assert np.isfinite(noise_beta_neg).all()


class TestParameterSetting:
    """Test parameter setting methods."""

    def test_set_parameter_methods(self) -> None:
        """Test parameter setting methods."""
        model = ArNoiseModel()

        # Test setting initial value
        model._set_initial("noise_alpha", 20.0)
        assert model.parameters.loc["noise_alpha", "initial"] == 20.0

        # Test setting min/max
        model._set_pmin("noise_alpha", 5.0)
        model._set_pmax("noise_alpha", 100.0)
        assert model.parameters.loc["noise_alpha", "pmin"] == 5.0
        assert model.parameters.loc["noise_alpha", "pmax"] == 100.0

        # Test setting vary
        model._set_vary("noise_alpha", False)
        assert not model.parameters.loc["noise_alpha", "vary"]

        # Test setting distribution
        model._set_dist("noise_alpha", "normal")
        assert model.parameters.loc["noise_alpha", "dist"] == "normal"


def test_irregular_time_steps(irregular_residual_series: Series) -> None:
    """Test noise models with irregular time steps."""
    ar_model = ArNoiseModel()

    # Set parameters
    ar_params = [10.0]  # alpha

    # Simulate noise
    ar_noise = ar_model.simulate(irregular_residual_series, ar_params)

    # Check properties
    assert len(ar_noise) == len(irregular_residual_series)

    # Calculate weights for AR model
    weights = ar_model.weights(irregular_residual_series, ar_params)
    assert len(weights) == len(irregular_residual_series)
