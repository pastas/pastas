"""This module contains utility functions for working with Pastas models."""

import logging
from logging import handlers
from platform import platform

# Type Hinting
from typing import Any, Optional, Tuple

from pandas import Timestamp

from pastas.typing import Model as ModelType
from pastas.typing import TimestampType

logger = logging.getLogger(__name__)


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
        A Logger-instance. Use pastas.logger to initialise the Logging instance that
        handles all logging throughout pastas, including all submodules and packages.
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
    """Set the log-level for Pastas.

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
    logger = logging.getLogger("pastas")
    logger.setLevel(level)


def remove_console_handler(logger: Optional[Any] = None) -> None:
    """Method to remove the console handler to the logger of Pastas.

    Parameters
    ----------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance that
        handles all logging throughout pastas, including all submodules and packages.
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
        handles all logging throughout pastas, including all submodules and packages.
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
                "User-provided name '%s' contains illegal character. Please "
                "remove '%s' from name."
            )
            if raise_error:
                logger.error(msg, name, char)
                raise Exception(msg % (name, char))
            else:
                logger.warning(msg, name, char)

    return name
