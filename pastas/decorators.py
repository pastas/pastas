from functools import wraps
from logging import getLogger
from typing import Any, Dict, Optional

from packaging.version import parse as parse_version

from pastas.typing import Function, TimestampType
from pastas.version import __version__

logger = getLogger(__name__)

USE_NUMBA = True
CURRENT_PASTAS_VERSION = parse_version(__version__)


def set_use_numba(b: bool) -> None:
    global USE_NUMBA
    USE_NUMBA = b


def set_parameter(function: Function) -> Function:
    @wraps(function)
    def _set_parameter(self, name: str, value: float, **kwargs):
        if name not in self.parameters.index:
            msg = "Parameter name %s does not exist, please choose from %s"
            logger.error(msg, name, self.parameters.index)
            raise KeyError(msg % (name, self.parameters.index))
        else:
            return function(self, name, value, **kwargs)

    return _set_parameter


def get_stressmodel(function: Function) -> Function:
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


def model_tmin_tmax(function: Function) -> Function:
    @wraps(function)
    def _model_tmin_tmax(
        self,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        *args,
        **kwargs,
    ):
        if tmin is None:
            tmin = self.ml.settings["tmin"]
        if tmax is None:
            tmax = self.ml.settings["tmax"]

        return function(self, tmin, tmax, *args, **kwargs)

    return _model_tmin_tmax


def PastasDeprecationWarning(remove_version: str, reason: str = "") -> Any:
    """Provide a warning or error when a Pastas class, method or function is deprecated.

    Logs a warning if the current Pastas version is lower than the version in which the
    class, function or method is removed. Raises a DeprecationWarning if the current
    Pastas version is higher than the version in which the class, function or method was
    removed.

    Parameters
    ----------
    remove_version: str
        The version in which the function or class will be removed.
    reason: str, optional
        The reason why the function or class is deprecated. Or provide a message
        that tells the user which alternative should be used.
    """

    def wrapper(obj: Any):
        name = obj.__name__

        def _function(*args, **kwargs):
            if CURRENT_PASTAS_VERSION < parse_version(remove_version):
                msg = (
                    "%s is deprecated and will be removed in Pastas version %s. "
                    % (name, remove_version)
                ) + reason
                logger.warning(msg)
            else:
                msg = (
                    "%s is deprecated and was removed in Pastas version %s. "
                    % (name, remove_version)
                ) + reason
                raise DeprecationWarning(msg)

            return obj(*args, **kwargs)

        return _function

    return wrapper


def deprecate_args_or_kwargs(
    name: str, remove_version: str, reason: str = "", force_raise: bool = False
):
    """Provide a warning or error when a function argument is deprecated.

    Parameters
    ----------
    name: str
        The name of the argument that is deprecated.
    remove_version: str
        The version in which the argument will be removed.
    reason: str, optional
        The reason why the argument is deprecated. Or provide a message that tells the
        user which alternative should be used.
    force_raise: bool, optional
        If True, raise a DeprecationWarning even if remove_version is still in the
        future. Default is False.
    """
    if (not force_raise) and (CURRENT_PASTAS_VERSION < parse_version(remove_version)):
        msg = (
            "The '%s' argument is deprecated and will be removed in Pastas version %s. "
            % (name, remove_version)
        ) + reason
        logger.warning(msg)
    else:
        if force_raise:
            msg = (
                "The %s argument is deprecated and will be removed in Pastas version %s. "
                % (name, remove_version)
            ) + reason
        else:
            msg = (
                "The %s argument is deprecated and was removed in Pastas version %s. "
                % (name, remove_version)
            ) + reason

        raise DeprecationWarning(msg)


def njit(function: Optional[Function] = None, parallel: bool = False) -> Function:
    def njit_decorator(f: Function):
        try:
            if not USE_NUMBA:
                return f
            else:
                from numba import njit

                fnumba = njit(f, parallel=parallel)
                return fnumba
        except ImportError:
            return f

    if function:
        return njit_decorator(function)

    return njit_decorator


def latexfun(
    function: Optional[Function] = None,
    identifiers: Optional[Dict[str, str]] = None,
    use_math_symbols: bool = True,
    use_raw_function_name: bool = False,
) -> Function:
    def latexify_decorator(f: Function) -> Function:
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
