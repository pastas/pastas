"""Tests for utility functions in pastas.utils."""

from typing import Any

import pandas as pd
import pytest

from pastas.utils import get_stress_tmin_tmax, validate_name


class TestGetStressTminTmax:
    """Test get_stress_tmin_tmax function."""

    def test_with_valid_model(self, ml_solved: Any) -> None:
        """Test with a valid model."""
        # Test function with real model from conftest
        tmin, tmax = get_stress_tmin_tmax(ml_solved)

        # Verify correct types
        assert isinstance(tmin, pd.Timestamp)
        assert isinstance(tmax, pd.Timestamp)

        # Verify tmin and tmax make sense relative to stress data
        for sm_name in ml_solved.stressmodels:
            for stress in ml_solved.stressmodels[sm_name].stresses:
                stress_tmin = stress.series_original.index.min()
                stress_tmax = stress.series_original.index.max()
                # tmin should be <= each stress's max time
                # tmax should be >= each stress's min time
                # This ensures there's at least some overlap in the time ranges
                assert tmin <= stress_tmax
                assert tmax >= stress_tmin

    def test_with_invalid_model(self) -> None:
        """Test with an invalid model type."""
        invalid_model = "Not a model"

        with pytest.raises(TypeError):
            get_stress_tmin_tmax(invalid_model)


class TestValidateName:
    """Test validate_name function."""

    def test_valid_name_linux(self) -> None:
        """Test with valid name on Linux."""
        name = "valid_name-123"
        result = validate_name(name)
        assert result == name

    def test_invalid_name_linux(self, caplog: Any) -> None:
        """Test with invalid name on Linux platform."""
        name = "invalid/name with space"

        result = validate_name(name)
        assert result == name
        assert "contains illegal character" in caplog.text

    def test_invalid_name_raise_error(self) -> None:
        """Test with invalid name and raise_error=True."""
        name = "invalid/name"

        with pytest.raises(Exception) as excinfo:
            validate_name(name, raise_error=True)

        assert "contains illegal character" in str(excinfo.value)

    def test_non_string_name(self) -> None:
        """Test with non-string name."""
        name = 12345
        result = validate_name(name)
        assert result == "12345"  # Converted to string
