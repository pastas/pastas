"""Tests for preventing direct manipulation of model attributes."""

import pandas as pd
import pytest

import pastas as ps


@pytest.fixture
def simple_model() -> ps.Model:
    """Create a simple model for testing."""
    import numpy as np

    dates = pd.date_range(start="2000-01-01", end="2005-12-31", freq="D")
    head = pd.Series(
        np.sin(np.linspace(0, 2 * np.pi * 10, len(dates))) + 5.0,
        index=dates,
        name="obs",
    )
    rain = pd.Series(np.abs(np.random.normal(0, 1, len(dates))), index=dates)

    ml = ps.Model(head, name="test_model")
    sm = ps.StressModel(rain, ps.Gamma(), name="recharge", settings="prec")
    ml.add_stressmodel(sm)
    return ml


class TestParametersProtection:
    """Test that parameters attribute is protected from direct manipulation."""

    def test_parameters_read_access(self, simple_model: ps.Model) -> None:
        """Test that we can read parameters."""
        params = simple_model.parameters
        assert isinstance(params, pd.DataFrame)
        assert not params.empty

    def test_parameters_assignment_raises_error(self, simple_model: ps.Model) -> None:
        """Test that assigning to parameters raises AttributeError."""
        with pytest.raises(
            AttributeError,
            match="Direct assignment to 'parameters' is not allowed",
        ):
            simple_model.parameters = pd.DataFrame()

    def test_parameters_returns_copy(self, simple_model: ps.Model) -> None:
        """Test that parameters returns a copy, not the original."""
        # Get original value
        original_value = simple_model.parameters.loc["recharge_A", "initial"]

        # Modify the returned DataFrame
        params = simple_model.parameters
        params.loc["recharge_A", "initial"] = 999.0

        # Check that the model's internal parameters are unchanged
        new_value = simple_model.parameters.loc["recharge_A", "initial"]
        assert new_value == original_value
        assert new_value != 999.0

    def test_set_parameter_still_works(self, simple_model: ps.Model) -> None:
        """Test that set_parameter method still works correctly."""
        original_value = simple_model.parameters.loc["recharge_A", "initial"]

        # Use the proper method to change parameters
        simple_model.set_parameter("recharge_A", initial=100.0)

        # Check that the change was applied
        new_value = simple_model.parameters.loc["recharge_A", "initial"]
        assert new_value == 100.0
        assert new_value != original_value


class TestSettingsProtection:
    """Test that settings attribute is protected from direct manipulation."""

    def test_settings_read_access(self, simple_model: ps.Model) -> None:
        """Test that we can read settings."""
        settings = simple_model.settings
        assert isinstance(settings, dict)
        assert "tmin" in settings
        assert "tmax" in settings

    def test_settings_assignment_raises_error(self, simple_model: ps.Model) -> None:
        """Test that assigning to settings raises AttributeError."""
        with pytest.raises(
            AttributeError,
            match="Direct assignment to 'settings' is not allowed",
        ):
            simple_model.settings = {}

    def test_settings_returns_copy(self, simple_model: ps.Model) -> None:
        """Test that settings returns a copy, not the original."""
        # Get original value
        original_tmin = simple_model.settings["tmin"]

        # Modify the returned dict
        settings = simple_model.settings
        settings["tmin"] = "2020-01-01"

        # Check that the model's internal settings are unchanged
        new_tmin = simple_model.settings["tmin"]
        assert new_tmin == original_tmin
        assert new_tmin != "2020-01-01"

    def test_solve_still_updates_settings(self, simple_model: ps.Model) -> None:
        """Test that solve() method still updates settings correctly."""
        # Solve the model
        simple_model.solve(tmin="2001-01-01", tmax="2004-12-31", report=False)

        # Check that settings were updated
        assert simple_model.settings["tmin"] is not None
        assert simple_model.settings["tmax"] is not None
        assert simple_model.settings["solver"] is not None
