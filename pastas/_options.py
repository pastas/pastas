"""Register and manage global settings for Pastas."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(slots=True)
class PastasOptions:
    """Central configuration storage for Pastas global settings."""

    cache: bool = False
    numba: bool = True
    parallel: bool = False
    timeseries: dict[str, OseriesSettingsDict | StressSettingsDict] = field(
        default_factory=lambda: {
            k: v.copy() for k, v in DEFAULT_TIMESERIES_SETTINGS.items()
        }
    )
    plotting: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Add plotting defaults
        self.plotting["max_gap"] = 50.0


options = PastasOptions()
