"""This module contains utility functions for working with Pastas models."""

import logging
from datetime import datetime
from logging import handlers
from platform import platform

# Type Hinting
from typing import Any, Optional, Tuple

from pandas import DatetimeIndex, Timestamp

from pastas.typing import ArrayLike
from pastas.typing import Model as ModelType
from pastas.typing import TimestampType

logger = logging.getLogger(__name__)


def excel_to_datetime(tindex: DatetimeIndex, freq="D") -> DatetimeIndex:
    raise DeprecationWarning("This function is deprecated and will be removed in v1.0.")


def matlab_datenum_to_datetime(datenum: float) -> datetime:
    raise DeprecationWarning("This function is deprecated and will be removed in v1.0.")


def datetime_to_matlab_datenum(tindex: DatetimeIndex) -> ArrayLike:
    raise DeprecationWarning("This function is deprecated and will be removed in v1.0.")


def get_stress_tmin_tmax(ml: ModelType) -> Tuple[TimestampType, TimestampType]:
    """Get the minimum and maximum time that all the stresses have data."""
    from pastas import Model

    tmin = Timestamp.min
    tmax = Timestamp.max
    if isinstance(ml, Model):
        for sm in ml.stressmodels:
            for st in ml.stressmodels[sm].stress:
                tmin = max((tmin, st.series_original.index.min()))
                tmax = min((tmax, st.series_original.index.max()))
    else:
        raise (TypeError("Unknown type {}".format(type(ml))))
    return tmin, tmax


def initialize_logger(
    logger: Optional[Any] = None, level: Optional[Any] = logging.INFO
) -> None:
    """Internal method to create a logger instance to log program output.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas,  including all submodules and packages.
    """
    if logger is None:
        logger = logging.getLogger("pastas")
    logger.setLevel(level)
    remove_file_handlers(logger)
    set_console_handler(logger)
    # add_file_handlers(logger)


def set_console_handler(
    logger: Optional[Any] = None,
    level: Optional[Any] = logging.INFO,
    fmt: str = "%(levelname)s: %(message)s",
) -> None:
    """Method to add a console handler to the logger of Pastas.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas,  including all submodules and packages.
    """
    if logger is None:
        logger = logging.getLogger("pastas")
    remove_console_handler(logger)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(fmt=fmt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_log_level(level: str) -> None:
    """Set the log-level of the console. This method is just a wrapper around
    set_console_handler.

    Parameters
    ----------
    level: str
        String with the level to log messages to the screen for. Options are: "INFO",
        "WARNING", and "ERROR".

    Examples
    --------

    >>> import pandas as ps
    >>> ps.set_log_level("ERROR")
    """
    set_console_handler(level=level)


def remove_console_handler(logger: Optional[Any] = None) -> None:
    """Method to remove the console handler to the logger of Pastas.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas, including all sub modules and packages.
    """
    if logger is None:
        logger = logging.getLogger("pastas")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


def add_file_handlers(
    logger: Optional[Any] = None,
    filenames: Tuple[str] = ("info.log", "errors.log"),
    levels: Tuple[Any] = (logging.INFO, logging.ERROR),
    maxBytes: int = 10485760,
    backupCount: int = 20,
    encoding: str = "utf8",
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt: str = "%y-%m-%d %H:%M",
) -> None:
    """Method to add file handlers in the logger of Pastas.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas, including all sub modules and packages.
    """
    if logger is None:
        logger = logging.getLogger("pastas")
    # create formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # create file handlers, set the level & formatter, and add it to the logger
    for filename, level in zip(filenames, levels):
        fh = handlers.RotatingFileHandler(
            filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def remove_file_handlers(logger: Optional[logging.Logger] = None) -> None:
    """Method to remove any file handlers in the logger of Pastas.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas, including all submodules and packages.
    """
    if logger is None:
        logger = logging.getLogger("pastas")
    for handler in logger.handlers:
        if isinstance(handler, handlers.RotatingFileHandler):
            logger.removeHandler(handler)


def validate_name(name: str, raise_error: bool = False) -> str:
    """Method to check user-provided names and log a warning if wrong.

    Parameters
    ----------
    name: str
        String with the name to check for illegal characters.
    raise_error: bool
        raise Exception error if illegal character is found, default is False which
        only logs a warning.

    Returns
    -------
    name: str
        Unchanged name string.
    """
    ilchar = ["/", "\\", " ", ".", "'", '"', "`"]
    if "windows" in platform().lower():
        ilchar += [
            "#",
            "%",
            "&",
            "@",
            "{",
            "}",
            "|",
            "$",
            "*",
            "<",
            ">",
            "?",
            "!",
            ":",
            "=",
            "+",
        ]

    name = str(name)
    for char in ilchar:
        if char in name:
            msg = (
                f"User-provided name '{name}' contains illegal character. Please "
                f"remove '{char}' from name."
            )
            if raise_error:
                raise Exception(msg)
            else:
                logger.warning(msg)

    return name


def get_sample(tindex, ref_tindex):
    raise DeprecationWarning("This method was moved to `pastas.ts.get_sample()`!")


def timestep_weighted_resample(series0, tindex):
    raise DeprecationWarning(
        "This method was moved to `pastas.ts.timestep_weighted_resample()`!"
    )


def timestep_weighted_resample_fase(series0, tindex):
    raise DeprecationWarning(
        "This method is deprecated and merged into "
        "`pastas.ts.timestep_weighted_resample()`!"
    )


def get_equidistant_series(series, freq, minimize_data_loss=False):
    raise DeprecationWarning(
        "This method was moved to `pastas.ts.get_equidistant_series()`!"
    )
