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
        assert not hasattr(options, "nonexistent")

    def test_repr(self):
        """Test __repr__ method."""
        from pastas._options import options

        repr_str = repr(options)
        assert "cache" in repr_str
        assert "numba" in repr_str
        assert "parallel" in repr_str

    def test_dir(self):
        """Test __dir__ method."""
        from pastas._options import options

        keys = dir(options)
        assert "cache" in keys
        assert "numba" in keys
        assert "parallel" in keys


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
