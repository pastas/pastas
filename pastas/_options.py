"""Register and manage global settings for Pastas."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PastasOptions:
    """Central configuration storage for Pastas global settings.

    Attributes
    ----------
    cache : bool
        Cache results of simulate calls which can improve performance.
    numba : bool
        Use numba. It is not recommended to turn off numba compilation
        unless for debugging.
    parallel : bool
        Use parallel computation in Pastas. Affects parallel numba-compiled
        functions and the Emcee solver.
    """

    cache: bool = False
    numba: bool = True
    parallel: bool = True


options = PastasOptions()
