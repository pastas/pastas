"""Tests for options API and options behavior in Pastas.

This module contains tests for the basic options functionality and options
access patterns.
"""

from dataclasses import is_dataclass

import pytest


class TestOptionsAPI:
    """Tests for the options API."""

    def test_attribute_style_get(self):
        """Test attribute-style access for getting values."""
        from pastas._options import options

        assert options.cache is False
        assert options.numba is True
        assert options.parallel is False
        assert hasattr(options, "timeseries")

    def test_attribute_style_set(self):
        """Test attribute-style access for setting values."""
        from pastas._options import options

        original_cache = options.cache
        try:
            options.cache = True
            assert options.cache is True
        finally:
            options.cache = original_cache

    def test_invalid_attribute_raises(self):
        """Test that accessing non-existent attribute raises AttributeError."""
        from pastas._options import options

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = options.nonexistent_key

    def test_set_invalid_attribute_raises(self):
        """Test that setting non-existent attribute raises AttributeError."""
        from pastas._options import options

        with pytest.raises(AttributeError, match="has no attribute"):
            options.nonexistent_key = 123

    def test_contains(self):
        """Test __contains__ method for attribute access."""
        from pastas._options import options

        # With slots=True, check for attribute existence
        assert hasattr(options, "cache")
        assert hasattr(options, "numba")
        assert hasattr(options, "parallel")
        assert hasattr(options, "timeseries")
        assert not hasattr(options, "nonexistent")

    def test_repr(self):
        """Test __repr__ method."""
        from pastas._options import options

        repr_str = repr(options)
        assert "cache" in repr_str
        assert "numba" in repr_str
        assert "parallel" in repr_str
        assert "timeseries" in repr_str

    def test_dir(self):
        """Test __dir__ method."""
        from pastas._options import options

        keys = dir(options)
        assert "cache" in keys
        assert "numba" in keys
        assert "parallel" in keys
        assert "timeseries" in keys


class TestGlobalSettings:
    """Tests for options access."""

    def test_options_exists(self):
        """Test that options exists in _config."""
        from pastas._options import options

        assert is_dataclass(options)

    def test_options_values(self):
        """Test that options has expected default values."""
        from pastas._options import options

        assert options.cache is False
        assert options.numba is True
        assert options.parallel is False
        assert hasattr(options, "timeseries")

    def test_options_timeseries_structure(self):
        """Test that timeseries settings have expected structure."""
        from pastas._options import options

        timeseries = options.timeseries
        assert isinstance(timeseries, dict)

        # Check expected stress types exist
        expected_types = {
            "oseries",
            "prec",
            "evap",
            "well",
            "waterlevel",
            "level",
            "flux",
            "quantity",
        }
        assert set(timeseries.keys()) == expected_types

        # Check prec has expected keys
        prec = timeseries["prec"]
        expected_prec_keys = {
            "sample_up",
            "sample_down",
            "fill_nan",
            "fill_before",
            "fill_after",
        }
        assert set(prec.keys()) == expected_prec_keys

        # Check oseries has expected keys
        oseries = timeseries["oseries"]
        expected_oseries_keys = {"fill_nan", "sample_down"}
        assert set(oseries.keys()) == expected_oseries_keys

    def test_options_mutability(self):
        """Test that options can be modified."""
        from pastas._options import options

        original_cache = options.cache
        try:
            options.cache = True
            assert options.cache is True
        finally:
            options.cache = original_cache


class TestOptionsGlobalSettingsLink:
    """Tests to verify options object behavior."""

    def test_options_reflects_changes(self):
        """Test that changes to options attributes are visible."""
        from pastas._options import options

        original_cache = options.cache
        try:
            options.cache = True
            assert options.cache is True
        finally:
            options.cache = original_cache

    def test_timeseries_is_dict(self):
        """Test that timeseries is a dictionary."""
        from pastas._options import options

        # Check it's a dict
        assert isinstance(options.timeseries, dict)
        # Check it has the expected keys
        assert "prec" in options.timeseries
        assert "evap" in options.timeseries
