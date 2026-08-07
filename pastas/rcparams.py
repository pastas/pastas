"""Module containing the default configuration parameters for Pastas time series.

Defines default settings for handling time series, e.g. for resampling and gap filling.

.. deprecated::
    The rcParams dictionary is deprecated. Use `pastas.options` instead for a more
    Pythonic API. For example:

    - Instead of `ps.rcParams["timeseries"]["prec"]`, use `ps.timeseries.settings["prec"]`

This module is maintained for backward compatibility only.
"""

from collections.abc import Mapping
from typing import Any

from pastas._options import options
from pastas.decorators import deprecate_class_func_or_method

reason = "Using ps.rcParams is deprecated. Use ps.options instead."


class _DeprecatedRcParams(Mapping[str, Any]):
    """A deprecated dictionary interface to Pastas settings.

    This class provides a dictionary-like interface to the central settings,
    but emits deprecation warnings when accessed for writes.

    Examples
    --------
    >>> import pastas as ps
    >>> # Read (allowed, no warning)
    >>> settings = ps.timeseries.settings
    >>> # Read (deprecated, will warn)
    >>> ps.rcParams["timeseries"] # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._data = options

    @deprecate_class_func_or_method(version="2.4.0", reason=reason)
    def __getitem__(self, key: str) -> Any:
        """Get a setting value by key."""
        return self._data[key]

    @deprecate_class_func_or_method(version="2.4.0", reason=reason)
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a setting value by key."""
        self._data[key] = value

    def __iter__(self):
        """Iterate over setting keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of settings."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation."""
        return repr(self._data)

    def keys(self):
        """Return setting keys."""
        return self._data.__dict__.keys()

    def items(self):
        """Return setting items."""
        return self._data.__dict__.items()

    def values(self):
        """Return setting values."""
        return self._data.__dict__.values()

    @deprecate_class_func_or_method(version="2.4.0", reason=reason)
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting with a default value."""
        return self._data.__dict__.get(key, default)

    @deprecate_class_func_or_method(version="2.4.0", reason=reason)
    def pop(self, key: str, *args: Any) -> Any:
        """Remove and return a setting."""
        return self._data.__dict__.pop(key, *args)

    @deprecate_class_func_or_method(version="2.4.0", reason=reason)
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update settings from a dictionary."""
        for key, value in dict(*args, **kwargs).items():
            setattr(self._data, key, value)

    def __contains__(self, value: Any) -> bool:
        """Check if a key exists."""
        return bool(value in self._data.__dict__)

    def __eq__(self, other: object) -> bool:
        """Check equality with another object."""
        if isinstance(other, Mapping):
            return dict(self._data.__dict__) == dict(other)
        return False


# Create the deprecated rcParams instance
rcParams: _DeprecatedRcParams = _DeprecatedRcParams()
