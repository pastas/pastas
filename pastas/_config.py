"""Central configuration storage for Pastas global settings.

This module contains the single source of truth for all Pastas configuration settings.
Both the options API and the legacy rcParams interface access these settings.
"""

from typing import Any

from pastas.typing import OseriesSettingsDict, StressSettingsDict

# Default settings for timeseries (migrated from rcparams.py)
DEFAULT_TIMESERIES_SETTINGS = {
    "oseries": OseriesSettingsDict(
        fill_nan="drop",
        sample_down="drop",
    ),
    "prec": StressSettingsDict(
        sample_up="bfill",
        sample_down="mean",
        fill_nan=0.0,
        fill_before="mean",
        fill_after="mean",
    ),
    "evap": StressSettingsDict(
        sample_up="bfill",
        sample_down="mean",
        fill_before="mean",
        fill_after="mean",
        fill_nan="interpolate",
    ),
    "well": StressSettingsDict(
        sample_up="bfill",
        sample_down="mean",
        fill_nan=0.0,
        fill_before=0.0,
        fill_after=0.0,
    ),
    "waterlevel": StressSettingsDict(
        sample_up="interpolate",
        sample_down="mean",
        fill_before="mean",
        fill_after="mean",
        fill_nan="interpolate",
    ),
    "level": StressSettingsDict(
        sample_up="interpolate",
        sample_down="mean",
        fill_before="mean",
        fill_after="mean",
        fill_nan="interpolate",
    ),
    "flux": StressSettingsDict(
        sample_up="bfill",
        sample_down="mean",
        fill_before="mean",
        fill_after="mean",
        fill_nan=0.0,
    ),
    "quantity": StressSettingsDict(
        sample_up="divide",
        sample_down="sum",
        fill_before="mean",
        fill_after="mean",
        fill_nan=0.0,
    ),
}

# Global settings dictionary
global_settings: dict[str, Any] = {
    # Global configuration settings
    "seed": 358183147,
    "cache": False,
    "numba": True,
    "parallel": False,
    # Timeseries settings (legacy, for backward compatibility)
    "timeseries": DEFAULT_TIMESERIES_SETTINGS,
}
