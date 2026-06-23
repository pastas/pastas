"""Tests for options API and global_settings behavior in Pastas.

This module contains tests for the basic options functionality and global_settings
access patterns.
"""

import pytest


class TestOptionsAPI:
    """Tests for the options API."""

    def test_attribute_style_get(self):
        """Test attribute-style access for getting values."""
        from pastas.options import options

        assert options.seed == 358183147
        assert options.cache is False
        assert options.numba is True
        assert options.parallel is False
        assert "timeseries" in options

    def test_attribute_style_set(self):
        """Test attribute-style access for setting values."""
        from pastas.options import options

        original_seed = options.seed
        try:
            options.seed = 999
            assert options.seed == 999
        finally:
            options.seed = original_seed

    def test_dict_style_get(self):
        """Test dict-style access for getting values."""
        from pastas.options import options

        assert options["seed"] == 358183147
        assert options["cache"] is False
        assert options["numba"] is True
        assert options["parallel"] is False

    def test_dict_style_set(self):
        """Test dict-style access for setting values."""
        from pastas.options import options

        original_seed = options.seed
        try:
            options["seed"] = 100
            assert options["seed"] == 100
            assert options.seed == 100
        finally:
            options.seed = original_seed

    def test_invalid_attribute_raises(self):
        """Test that accessing non-existent attribute raises AttributeError."""
        from pastas.options import options

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = options.nonexistent_key

    def test_invalid_key_raises(self):
        """Test that accessing non-existent key raises KeyError."""
        from pastas.options import options

        with pytest.raises(KeyError, match="has no key"):
            _ = options["nonexistent_key"]

    def test_set_invalid_attribute_raises(self):
        """Test that setting non-existent attribute raises AttributeError."""
        from pastas.options import options

        with pytest.raises(AttributeError, match="has no attribute"):
            options.nonexistent_key = 123

    def test_set_invalid_key_raises(self):
        """Test that setting non-existent key raises KeyError."""
        from pastas.options import options

        with pytest.raises(KeyError, match="has no key"):
            options["nonexistent_key"] = 123

    def test_contains(self):
        """Test __contains__ method."""
        from pastas.options import options

        assert "seed" in options
        assert "cache" in options
        assert "numba" in options
        assert "parallel" in options
        assert "timeseries" in options
        assert "nonexistent" not in options

    def test_iteration(self):
        """Test iteration over options."""
        from pastas.options import options

        keys = set(options)
        assert "seed" in keys
        assert "cache" in keys
        assert "numba" in keys
        assert "parallel" in keys
        assert "timeseries" in keys

    def test_len(self):
        """Test __len__ method."""
        from pastas.options import options

        # Should have: seed, cache, numba, parallel, timeseries
        assert len(options) == 5

    def test_repr(self):
        """Test __repr__ method."""
        from pastas.options import options

        repr_str = repr(options)
        assert "seed" in repr_str
        assert "cache" in repr_str

    def test_dir(self):
        """Test __dir__ method."""
        from pastas.options import options

        keys = dir(options)
        assert "seed" in keys
        assert "cache" in keys


class TestGlobalSettings:
    """Tests for global_settings access."""

    def test_global_settings_exists(self):
        """Test that global_settings exists in _config."""
        from pastas._config import global_settings

        assert isinstance(global_settings, dict)

    def test_global_settings_has_expected_keys(self):
        """Test that global_settings has expected keys."""
        from pastas._config import global_settings

        expected_keys = {"seed", "cache", "numba", "parallel", "timeseries"}
        assert set(global_settings.keys()) == expected_keys

    def test_global_settings_values(self):
        """Test that global_settings has expected default values."""
        from pastas._config import global_settings

        assert global_settings["seed"] == 358183147
        assert global_settings["cache"] is False
        assert global_settings["numba"] is True
        assert global_settings["parallel"] is False
        assert "timeseries" in global_settings

    def test_global_settings_timeseries_structure(self):
        """Test that timeseries settings have expected structure."""
        from pastas._config import global_settings

        timeseries = global_settings["timeseries"]
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

    def test_global_settings_mutability(self):
        """Test that global_settings can be modified."""
        from pastas._config import global_settings

        original_seed = global_settings["seed"]
        try:
            global_settings["seed"] = 999
            assert global_settings["seed"] == 999
        finally:
            global_settings["seed"] = original_seed


class TestOptionsGlobalSettingsLink:
    """Tests to verify options and global_settings are linked."""

    def test_options_reflects_global_settings_changes(self):
        """Test that changes to global_settings are visible through options."""
        from pastas._config import global_settings
        from pastas.options import options

        original_seed = global_settings["seed"]
        try:
            global_settings["seed"] = 12345
            assert options.seed == 12345
            assert options["seed"] == 12345
        finally:
            global_settings["seed"] = original_seed

    def test_global_settings_reflects_options_changes(self):
        """Test that changes to options are visible through global_settings."""
        from pastas._config import global_settings
        from pastas.options import options

        original_seed = options.seed
        try:
            options.seed = 54321
            assert global_settings["seed"] == 54321
        finally:
            options.seed = original_seed

    def test_timeseries_link(self):
        """Test that timeseries settings are linked between options and global_settings."""
        from pastas._config import global_settings
        from pastas.options import options

        # Check they reference the same object
        assert options.timeseries is global_settings["timeseries"]
        assert options["timeseries"] is global_settings["timeseries"]
