"""Module containing decorators and utility functions for Pastas models.

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

from pastas._config import global_settings
from pastas.version import __version__

try:
    from cachetools import cachedmethod

    CACHETOOLS_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    CACHETOOLS_AVAILABLE = False

logger = getLogger(__name__)

# Keep these for backward compatibility but they now delegate to central settings
# These will be removed in a future version 2.3.0
USE_NUMBA = True
USE_CACHE = False
CURRENT_PASTAS_VERSION = parse_version(__version__)


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The set_use_numba function is deprecated. Use ps.options.numba = True/False instead.",
)
def set_use_numba(b: bool) -> None:
    """Enable or disable the use of Numba JIT compilation.

    .. deprecated::
        Use `pastas.options.numba = True/False` instead.
    """
    global_settings["numba"] = b
    # Update the module-level global for backward compatibility
    global USE_NUMBA
    USE_NUMBA = b


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The get_use_numba function is deprecated. Use ps.options.numba instead.",
)
def get_use_numba() -> bool:
    """Check if Numba JIT compilation is enabled.

    .. deprecated::
        Use `pastas.options.numba` instead.
    """
    return global_settings["numba"]


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The set_use_cache function is deprecated. Use ps.options.cache = True/False instead.",
)
def set_use_cache(b: bool) -> None:
    """Enable or disable the use of caching with cachetools.

    When caching is enabled, the results of simulate() calls are stored in a cache
    to speed up repeated calls with the same parameters. This requires the cachetools
    package to be installed and the USE_CACHE variable to be set to True.

    .. deprecated::
        Use `pastas.options.cache = True/False` instead.
    """
    if b and not CACHETOOLS_AVAILABLE:
        logger.error(
            "Cannot enable caching: cachetools is not installed. "
            "Install with: pip install cachetools"
        )
        return
    global_settings["cache"] = b
    # Update the module-level global for backward compatibility
    global USE_CACHE
    USE_CACHE = b


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The get_use_cache function is deprecated. Use ps.options.cache instead.",
)
def get_use_cache() -> bool:
    """Check if caching with cachetools is enabled.

    .. deprecated::
        Use `pastas.options.cache` instead.
    """
    return global_settings["cache"]


def set_parameter(function: Callable) -> Callable:
    """Validate and set parameter values.

    This decorator checks if the parameter name exists in the parameters DataFrame
    before calling the wrapped function.

    Parameters
    ----------
    function : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function with parameter validation.
    """

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
    """Validate and retrieve stressmodel by name.

    This decorator checks if the stressmodel name exists before calling the wrapped function.

    Parameters
    ----------
    function : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function with stressmodel validation.
    """

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
    """Use model tmin and tmax settings as default values.

    This decorator uses the model's tmin and tmax settings if they are not provided.

    Parameters
    ----------
    function : Callable
        The function to wrap.

    Returns
    -------
    Callable
        The wrapped function with default tmin/tmax from model settings.
    """

    @wraps(function)
    def _model_tmin_tmax(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        *args,
        **kwargs,
    ):
        tmin = self.ml.settings["tmin"] if tmin is None else Timestamp(tmin)
        tmax = self.ml.settings["tmax"] if tmax is None else Timestamp(tmax)

        return function(self, tmin, tmax, *args, **kwargs)

    return _model_tmin_tmax


def PastasDeprecationWarning(version: str, reason: str = "") -> Any:
    """Provide a warning or error when a Pastas class, method or function is deprecated.

    This decorator manages deprecation of classes, functions, or methods across Pastas versions.
    The behavior depends on the current version:

    - If current version < version: logs a warning and allows execution to continue
    - If current version >= version: raises AttributeError which indicates
    that it can be removed from the codebase entirely in the (near) future.

    Parameters
    ----------
    version: str
        The version in which the function, method or class raises an AttributeError.
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
                    f"from Pastas version >= {VERSION}. {reason}"
                )
                warn(message=msg, category=DeprecationWarning)
            else:
                msg = (
                    f"Module has no attribute '{name}'. "
                    f"{name} is deprecated and is not available since"
                    f" Pastas version {VERSION}. {reason}"
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
    removed from the codebase entirely in the (near) future.

    Parameters
    ----------
    name: str
        The name of the argument that is deprecated.
    version: str
        The version in which using the argument will raise a TypeError.
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
            f" from Pastas version >= {VERSION}. {reason}"
        )
        warn(message=msg, category=DeprecationWarning)
    else:
        msg = (
            f"Got an unexpected keyword argument {name}. "
            f"The {name} argument is deprecated and is not available"
            f" since Pastas version {VERSION}. {reason}"
        )
        raise TypeError(msg)


def njit(function: Callable | None = None, **kwargs) -> Callable:
    """Apply numba's njit to a function if numba is available.

    Parameters
    ----------
    function : callable, optional
        The function to decorate.
    **kwargs
        Additional keyword arguments passed to numba.njit.

    Returns
    -------
    callable
        The decorated function, or the original function if numba is not available.
    """

    def njit_decorator(f: Callable) -> Callable:
        try:
            if not global_settings["numba"]:
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


@PastasDeprecationWarning(
    version="2.0.0",
    reason="latexify was archived and is no longer maintained. This decorator will be removed in a future release.",
)
def latexfun(**kwargs) -> None:
    """Use deprecated latexify functionality.

    This decorator is deprecated and will be removed in a future release.
    """
    pass


def conditional_cachedmethod(cache_getter):
    """Conditionally cache a method using cachetools.cachedmethod.

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
            if global_settings["cache"]:
                return cached_func.__get__(self, type(self))(*args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The temporarily_disable_cache context manager is deprecated. Use ps.options.cache = False directly instead.",
)
@contextmanager
def temporarily_disable_cache():
    """Context manager to temporarily disable caching.

    Examples
    --------
    To temporarily disable the cache (if it is currently active)::

        with ps.temporarily_disable_cache():
            # Caching is disabled within this block
            ml.simulate()

    .. deprecated::
        Use `ps.options.cache` directly instead.
    """
    original_state = global_settings["cache"]
    global_settings["cache"] = False
    # Update module-level global for backward compatibility
    global USE_CACHE
    original_use_cache = USE_CACHE
    USE_CACHE = False
    try:
        yield
    finally:
        global_settings["cache"] = original_state
        USE_CACHE = original_use_cache


@PastasDeprecationWarning(
    version="2.3.0",
    reason="The temporarily_enable_cache context manager is deprecated. Use ps.options.cache = True directly instead.",
)
@contextmanager
def temporarily_enable_cache():
    """Context manager to temporarily enable caching.

    Examples
    --------
    >>> with ps.temporarily_enable_cache():
    ...     # Caching is enabled within this block
    ...     ml.simulate()

    .. deprecated::
        Use `ps.options.cache` directly instead.
    """
    original_state = global_settings["cache"]
    global_settings["cache"] = True
    # Update module-level global for backward compatibility
    global USE_CACHE
    original_use_cache = USE_CACHE
    USE_CACHE = True
    try:
        yield
    finally:
        global_settings["cache"] = original_state
        USE_CACHE = original_use_cache
