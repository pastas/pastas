from functools import wraps
from logging import getLogger

from typing import Optional, Dict
from pastas.typing import Function, TimestampType

logger = getLogger(__name__)

USE_NUMBA = True


def set_use_numba(b: bool) -> None:
    global USE_NUMBA
    USE_NUMBA = b


def set_parameter(function: Function) -> Function:
    @wraps(function)
    def _set_parameter(self, name: str, value: float, **kwargs):
        if name not in self.parameters.index:
            logger.error(
                "Parameter name %s does not exist, please choose from %s",
                name,
                self.parameters.index,
            )
        else:
            return function(self, name, value, **kwargs)

    return _set_parameter


def get_stressmodel(function: Function) -> Function:
    @wraps(function)
    def _get_stressmodel(self, name: str, **kwargs):
        if name not in self.stressmodels.keys():
            logger.error(
                "The stressmodel name you provided is not in the stressmodels dict. "
                "Please select from the following list: %s",
                self.stressmodels.keys(),
            )
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
        **kwargs
    ):
        if tmin is None:
            tmin = self.ml.settings["tmin"]
        if tmax is None:
            tmax = self.ml.settings["tmax"]

        return function(self, tmin, tmax, *args, **kwargs)

    return _model_tmin_tmax


def PastasDeprecationWarning(function: Function) -> Function:
    @wraps(function)
    def _function(*args, **kwargs):
        logger.warning(
            "Method is deprecated and will be removed in Pastas version 1.0."
        )
        return function(*args, **kwargs)

    return _function


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
