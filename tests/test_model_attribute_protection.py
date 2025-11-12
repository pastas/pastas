"""Tests for preventing direct manipulation of model attributes."""

import pandas as pd
import pytest

import pastas as ps


class TestParametersProtection:
    """Test that parameters attribute is protected from direct manipulation."""

    def test_parameters_read_access(self, ml_recharge: ps.Model) -> None:
        """Test that we can read parameters."""
        params = ml_recharge.parameters
        assert isinstance(params, pd.DataFrame)
        assert not params.empty

    def test_parameters_assignment_raises_error(self, ml_recharge: ps.Model) -> None:
        """Test that assigning to parameters raises AttributeError."""
        with pytest.raises(
            AttributeError,
            match="Direct assignment to 'parameters' is not allowed",
        ):
            ml_recharge.parameters = pd.DataFrame()

    def test_parameters_returns_copy(self, ml_recharge: ps.Model) -> None:
        """Test that parameters returns a copy, not the original."""
        # Get original value
        original_value = ml_recharge.parameters.loc["rch_A", "initial"]

        # Modify the returned DataFrame
        params = ml_recharge.parameters
        params.loc["rch_A", "initial"] = 999.0

        # Check that the model's internal parameters are unchanged
        new_value = ml_recharge.parameters.loc["rch_A", "initial"]
        assert new_value == original_value
        assert new_value != 999.0

    def test_set_parameter_still_works(self, ml_recharge: ps.Model) -> None:
        """Test that set_parameter method still works correctly."""
        original_value = ml_recharge.parameters.loc["rch_A", "initial"]

        # Use the proper method to change parameters
        ml_recharge.set_parameter("rch_A", initial=100.0)

        # Check that the change was applied
        new_value = ml_recharge.parameters.loc["rch_A", "initial"]
        assert new_value == 100.0
        assert new_value != original_value


class TestSettingsProtection:
    """Test that settings attribute is protected from direct manipulation."""

    def test_settings_read_access(self, ml_recharge: ps.Model) -> None:
        """Test that we can read settings."""
        settings = ml_recharge.settings
        assert isinstance(settings, dict)
        assert "tmin" in settings
        assert "tmax" in settings

    def test_settings_assignment_raises_error(self, ml_recharge: ps.Model) -> None:
        """Test that assigning to settings raises AttributeError."""
        with pytest.raises(
            AttributeError,
            match="Direct assignment to 'settings' is not allowed",
        ):
            ml_recharge.settings = {}

    def test_settings_returns_copy(self, ml_recharge: ps.Model) -> None:
        """Test that settings returns a copy, not the original."""
        # Get original value
        original_tmin = ml_recharge.settings["tmin"]

        # Modify the returned dict
        settings = ml_recharge.settings
        settings["tmin"] = "2020-01-01"

        # Check that the model's internal settings are unchanged
        new_tmin = ml_recharge.settings["tmin"]
        assert new_tmin == original_tmin
        assert new_tmin != "2020-01-01"

    def test_solve_still_updates_settings(self, ml_recharge: ps.Model) -> None:
        """Test that solve() method still updates settings correctly."""
        # Solve the model
        ml_recharge.solve(tmin="2011-01-01", tmax="2014-12-31", report=False)

        # Check that settings were updated
        assert ml_recharge.settings["tmin"] is not None
        assert ml_recharge.settings["tmax"] is not None
        assert ml_recharge.settings["solver"] is not None
