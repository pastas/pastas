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

    seed: int | None = 358183147
    cache: bool = False
    numba: bool = True
    parallel: bool = False
    timeseries: dict[str, OseriesSettingsDict | StressSettingsDict] = field(
        default_factory=lambda: {
             k: v.copy() for k, v in DEFAULT_TIMESERIES_SETTINGS.items()
         }
    )

    def __getitem__(self, key: str) -> Any:
        """Get the value of a setting in the PastasOptions dataclass as a dictionary."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(
                f"Invalid setting: '{key}'. Available: {list(self.__slots__)}"
            )

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value of a setting in the PastasOptions dataclass as a dictionary."""
        if not hasattr(self, key):
            raise KeyError(
                f"Invalid setting: '{key}'. Available: {list(self.__slots__)}"
            )
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Return True if the PastasOptions dataclass has the specified setting."""
        return hasattr(self, key)

    def __iter__(self):
        """Return an iterator over the settings in the PastasOptions dataclass."""
        return iter(self.__slots__)

    def __len__(self) -> int:
        """Return the number of settings in the PastasOptions dataclass."""
        return len(self.__slots__)

    def keys(self):
        """Return the keys of the PastasOptions dataclass."""
        return self.__slots__


options = PastasOptions()
