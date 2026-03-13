"""This module contains decorators and utility functions for Pastas models.

Includes decorators for caching, configuring global settings, deprecation warnings,
and other convenient methods for handling time, numba compiled code, etc.
"""

from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from logging import getLogger
from typing import Any
from warnings import warn
from packaging.version import parse as parse_version
from pandas import Timestamp

from pastas.version import __version__

try:
    from cachetools import cachedmethod

    CACHETOOLS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    CACHETOOLS_AVAILABLE = False

logger = getLogger(__name__)

USE_NUMBA = True
USE_CACHE = False
CURRENT_PASTAS_VERSION = parse_version(__version__)


def set_use_numba(b: bool) -> None:
    """Enable or disable the use of Numba JIT compilation."""
    global USE_NUMBA
    USE_NUMBA = b


def get_use_numba() -> bool:
    """Check if Numba JIT compilation is enabled."""
    global USE_NUMBA
    return USE_NUMBA


def set_use_cache(b: bool) -> None:
    """Enable or disable the use of caching with cachetools.

    When caching is enabled, the results of simulate() calls are stored in a cache
    to speed up repeated calls with the same parameters. This requires the cachetools
    package to be installed and the USE_CACHE variable to be set to True.
    """
    global USE_CACHE
    if b and not CACHETOOLS_AVAILABLE:
        logger.error(
            "Cannot enable caching: cachetools is not installed. "
            "Install with: pip install cachetools"
        )
        return
    USE_CACHE = b


def get_use_cache() -> bool:
    """Check if caching with cachetools is enabled."""
    global USE_CACHE
    return USE_CACHE


def set_parameter(function: Callable) -> Callable:
    @wraps(function)
    def _set_parameter(self, name: str, value: float, **kwargs):
        if name not in self.parameters.index:
            msg = "Parameter name %s does not exist, please choose from %s"
            logger.error(msg, name, self.parameters.index)
            raise KeyError(msg % (name, self.parameters.index))
        else:
            return function(self, name, value, **kwargs)

    return _set_parameter


def get_stressmodel(function: Callable) -> Callable:
    @wraps(function)
    def _get_stressmodel(self, name: str, **kwargs):
        if name not in self.stressmodels.keys():
            msg = (
                "The stressmodel name you provided is not in the stressmodels dict. "
                "Please select from the following list: %s"
            )
            logger.error(msg, self.stressmodels.keys())
            raise KeyError(msg % self.stressmodels.keys())
        else:
            return function(self, name, **kwargs)

    return _get_stressmodel


def model_tmin_tmax(function: Callable) -> Callable:
    @wraps(function)
    def _model_tmin_tmax(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        *args,
        **kwargs,
    ):
        if tmin is None:
            tmin = self.ml.settings["tmin"]
        if tmax is None:
            tmax = self.ml.settings["tmax"]

        return function(self, tmin, tmax, *args, **kwargs)

    return _model_tmin_tmax


def PastasDeprecationWarning(version: str, reason: str = "") -> Any:
    """Provide a warning or error when a Pastas class, method or function is deprecated.

    This decorator manages deprecation of classes, functions, or methods across Pastas versions.
    The behavior depends on the current version:

    - If current version < version: logs a warning and allows execution to continue
    - If current version >= version: raises AttributeError which indicates
    that it can be removed from the codebase entirely

    Parameters
    ----------
    version: str
        The version in which the function or class will be removed from the Pastas codebase
        and raises an AttributeError.
    reason: str, optional
        The reason why the function or class is deprecated, or a message directing users
        to an alternative. Default is an empty string.

    Returns
    -------
    callable
        A decorator that wraps the target class or function.
    """

    def wrapper(obj: Any):
        name = obj.__name__

        def _function(*args, **kwargs):
            VERSION = parse_version(version)
            if CURRENT_PASTAS_VERSION < VERSION:
                msg = (
                    f"{name} is deprecated and will not be available "
                    f"from Pastas version >={VERSION}. {reason}"
                )
                warn(message=msg, category=DeprecationWarning)
            else:
                msg = (
                    f"module has no attribute '{name}'",
                    f"{name} is deprecated and is not available since"
                    f" Pastas version {VERSION}. {reason}",
                )
                raise AttributeError(msg)

            return obj(*args, **kwargs)

        return _function

    return wrapper


def deprecate_args_or_kwargs(name: str, version: str, reason: str = "") -> None:
    """Provide a warning or error when a function argument is deprecated.

    This function raises errors or warnings based on the current Pastas version and the
    deprecation timeline. The behavior is:

    - If current version < version: logs a warning
    - If current version >= version: raises TypeError which indicates that it can be
    removed from the codebase entirely

    Parameters
    ----------
    name: str
        The name of the argument that is deprecated.
    version: str
        The version in which the argument will be removed and TypeError will be raised.
    reason: str, optional
        The reason why the argument is deprecated, or a message directing users to
        an alternative. Default is an empty string.

    Raises
    ------
    DeprecationWarning
        If current version < version and the argument is used.
    TypeError
        If current version >= version and the argument is used.
    """
    VERSION = parse_version(version)
    if CURRENT_PASTAS_VERSION < VERSION:
        msg = (
            f"The {name} argument is deprecated and will not be available"
            f" from Pastas version >={VERSION}. {reason}"
        )
        warn(message=msg, category=DeprecationWarning)
    else:
        msg = (
            f"got an unexpected keyword argument {name}"
            f"The {name} argument is deprecated and is not available"
            f" since Pastas version {VERSION}. {reason}"
        )
        raise TypeError(msg)


def njit(function: Callable | None = None, **kwargs) -> Callable:
    def njit_decorator(f: Callable) -> Callable:
        try:
            if not USE_NUMBA:
                return f
            else:
                from numba import njit

                fnumba = njit(f, **kwargs)
                return fnumba
        except ImportError:
            return f

    if function:
        return njit_decorator(function)

    return njit_decorator


def latexfun(
    function: Callable | None = None,
    identifiers: dict[str, str] | None = None,
    use_math_symbols: bool = True,
    use_raw_function_name: bool = False,
) -> Callable:
    def latexify_decorator(f: Callable) -> Callable:
        try:
            import latexify

            flatex = latexify.function(
                f,
                identifiers=identifiers,
                use_math_symbols=use_math_symbols,
                use_raw_function_name=use_raw_function_name,
            )
            return flatex
        except ImportError:
            return f

    if function:
        return latexify_decorator(function)

    return latexify_decorator


def conditional_cachedmethod(cache_getter):
    """Decorator to conditionally cache a method using cachetools.cachedmethod.

    This decorator checks the global USE_CACHE flag and only applies caching when
    both cachetools is available and caching is enabled.

    Parameters
    ----------
    cache_getter : callable
        Function that returns the cache object from self (e.g., lambda self: self._cache)
    """

    def decorator(func):
        if not CACHETOOLS_AVAILABLE:
            # No cachetools available - just return the original function
            return func

        # Create the cached version once at decoration time
        cached_func = cachedmethod(cache_getter)(func)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if USE_CACHE:
                return cached_func(self, *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def temporarily_disable_cache():
    """Context manager to temporarily disable caching.

    Examples
    --------
    To temporarily disable the cache (if it is currently active)::

        with ps.temporarily_disable_cache():
            # Caching is disabled within this block
            ml.simulate()
    """
    global USE_CACHE
    original_state = USE_CACHE
    USE_CACHE = False
    try:
        yield
    finally:
        USE_CACHE = original_state


@contextmanager
def temporarily_enable_cache():
    """Context manager to temporarily enable caching.

    Examples
    --------
    >>> with ps.temporarily_enable_cache():
    ...     # Caching is enabled within this block
    ...     ml.simulate()
    """
    global USE_CACHE
    original_state = USE_CACHE
    USE_CACHE = True
    try:
        yield
    finally:
        USE_CACHE = original_state
