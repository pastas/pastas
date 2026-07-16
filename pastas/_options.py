"""Register and manage global settings for Pastas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PastasOptions:
    """Central configuration storage for Pastas global settings."""

    cache: bool = False
    numba: bool = True
    parallel: bool = False


options = PastasOptions()
