"""Module providing pandas-like options API for Pastas global settings.

This is the recommended way to access and modify Pastas configuration settings.

Examples
--------
>>> import pastas as ps
>>> ps.options.seed = 42  # Set global random seed (attribute-style)
>>> ps.options.cache = True  # Enable caching
>>> print(ps.options.numba)  # Get numba setting
>>> print(ps.options.parallel)  # Get parallel setting
>>> ps.options["seed"] = 100  # Dict-style access also works
>>> print(ps.options["seed"])

Notes
-----
This API provides attribute-style and dict-style access to the central
settings storage. For backward compatibility, the legacy `ps.rcParams`
dictionary is also available but its use is discouraged.
"""

from __future__ import annotations

from typing import Any

from pastas._config import global_settings


class OptionsContainer:
    """Container for Pastas global options.

    This class provides attribute-style and dict-style access to global settings,
    similar to pandas.options.

    All settings are stored in a central dictionary and can be accessed
    either through this attribute-style API, dict-style API, or through the legacy
    rcParams dictionary (which is deprecated for writes).

    Examples
    --------
    Attribute-style access:

    >>> import pastas as ps
    >>> ps.options.seed = 42
    >>> print(ps.options.seed)
    42
    >>> ps.options.cache = True
    >>> print(ps.options.cache)
    True

    Dict-style access:

    >>> ps.options["seed"] = 42
    >>> print(ps.options["seed"])
    42

    Notes
    -----
    The following settings are available:

    - seed : int | None
        Global random seed for reproducibility. If None, no seed is set.
    - cache : bool
        Global enable/disable caching for simulate() calls.
        Requires cachetools to be installed.
    - numba : bool
        Global enable/disable Numba JIT compilation.
    - parallel : bool
        Global enable/disable parallelization.
    - timeseries : dict
        Timeseries settings (legacy, read-only recommended).
    """

    def __init__(self) -> None:
        """Initialize the options container."""
        # Use object.__setattr__ to avoid triggering our custom __setattr__
        object.__setattr__(self, "_data", global_settings)

    def __getattr__(self, name: str) -> Any:
        """Get a setting value by attribute name.

        Parameters
        ----------
        name : str
            The name of the setting to retrieve.

        Returns
        -------
        Any
            The value of the setting.

        Raises
        ------
        AttributeError
            If the setting does not exist.
        """
        if name in self._data:
            return self._data[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'. "
            f"Available settings: {list(self._data.keys())}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        """Set a setting value by attribute name.

        Parameters
        ----------
        name : str
            The name of the setting to modify.
        value : Any
            The new value for the setting.

        Raises
        ------
        AttributeError
            If the setting does not exist.
        """
        if name.startswith("_"):
            # Allow setting internal attributes
            object.__setattr__(self, name, value)
        elif name in self._data:
            self._data[name] = value
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'. "
                f"Available settings: {list(self._data.keys())}"
            )

    def __getitem__(self, key: str) -> Any:
        """Get a setting value by key.

        Parameters
        ----------
        key : str
            The key of the setting to retrieve.

        Returns
        -------
        Any
            The value of the setting.

        Raises
        ------
        KeyError
            If the setting does not exist.
        """
        if key in self._data:
            return self._data[key]
        raise KeyError(
            f"'{type(self).__name__}' object has no key '{key}'. "
            f"Available settings: {list(self._data.keys())}"
        )

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a setting value by key.

        Parameters
        ----------
        key : str
            The key of the setting to modify.
        value : Any
            The new value for the setting.

        Raises
        ------
        KeyError
            If the setting does not exist.
        """
        if key in self._data:
            self._data[key] = value
        else:
            raise KeyError(
                f"'{type(self).__name__}' object has no key '{key}'. "
                f"Available settings: {list(self._data.keys())}"
            )

    def __contains__(self, key: str) -> bool:
        """Check if a key exists."""
        return key in self._data

    def __iter__(self):
        """Iterate over setting keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of settings."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the options.

        Returns
        -------
        str
            A string representation showing all settings.
        """
        return repr(self._data)

    def __dir__(self) -> list[str]:
        """Return a list of available setting names.

        Returns
        -------
        list[str]
            List of setting names plus internal attributes.
        """
        return list(self._data.keys()) + ["_data"]

    def __getstate__(self) -> dict[str, Any]:
        """Support for pickling."""
        return self._data

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support for unpickling."""
        object.__setattr__(self, "_data", state)


# Create singleton instance
options = OptionsContainer()
