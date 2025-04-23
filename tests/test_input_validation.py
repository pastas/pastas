"""Tests for input validation across Pastas."""

import numpy as np
import pandas as pd
import pytest

import pastas as ps
from pastas.timeseries import TimeSeries


class TestInputValidation:
    """Test input validation for various Pastas components."""

    def test_invalid_time_series_inputs(self) -> None:
        """Test handling of invalid time series inputs."""
        # Test with empty series
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            TimeSeries(empty_series)

        # Test with series containing only NaNs
        nan_series = pd.Series([np.nan, np.nan], index=pd.date_range("2000", periods=2))
        with pytest.raises(ValueError):
            TimeSeries(nan_series)

    def test_invalid_model_parameters(self, ml_noisemodel: ps.Model) -> None:
        """Test handling of invalid model parameters."""
        # Use the ml_solved fixture from conftest.py
        ml = ml_noisemodel

        # Test invalid parameter name
        with pytest.raises(KeyError):
            ml.set_parameter("nonexistent_parameter", initial=1.0)

        # Test inconsistent bounds
        param_name = "constant_d"
        with pytest.raises(ValueError):
            # Setting pmin > pmax should raise error
            ml.set_parameter(param_name, initial=7.5, pmin=10.0, pmax=5.0)

        # Test invalid parameter value type
        with pytest.raises(TypeError):
            ml.set_parameter(param_name, initial="not_a_number")

        # Test setting invalid bounds types
        with pytest.raises(TypeError):
            ml.set_parameter(param_name, initial=5.0, pmin="invalid")

        with pytest.raises(TypeError):
            ml.set_parameter(param_name, initial=5.0, pmax="invalid")

        # Test that initial value respects bounds
        with pytest.raises(ValueError):
            ml.set_parameter(param_name, initial=0.0, pmin=1.0, pmax=5.0)

        with pytest.raises(ValueError):
            ml.set_parameter(param_name, initial=10.0, pmin=1.0, pmax=5.0)

    def test_invalid_solve_parameters(self) -> None:
        """Test handling of invalid solve parameters."""
        dates = pd.date_range("2000", "2001", freq="D")
        head = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)
        ml = ps.Model(head)

        # Test invalid tmin/tmax
        with pytest.raises(ValueError):
            ml.solve(tmin="2002", tmax="2001")

        # Test invalid frequency
        with pytest.raises(ValueError):
            ml.solve(freq="invalid_freq")

    def test_incompatible_time_ranges(self) -> None:
        """Test handling of incompatible time ranges between series."""
        # Create series with non-overlapping time ranges
        dates1 = pd.date_range("2000", "2001", freq="D")
        dates2 = pd.date_range("2002", "2003", freq="D")

        head = pd.Series(np.random.normal(0, 1, len(dates1)), index=dates1)
        stress = pd.Series(np.random.normal(0, 1, len(dates2)), index=dates2)

        ml = ps.Model(head)
        sm = ps.StressModel(stress, rfunc=ps.Exponential(), name="stress")

        # Adding the stress model should work but produce a warning about no overlap
        ml.add_stressmodel(sm)

        # Solving should raise an error about time series extension
        with pytest.raises(ValueError, match="cannot be extended into past"):
            ml.solve()
