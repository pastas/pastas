"""This module contains utility functions for working with Pastas models."""

import logging
from datetime import datetime, timedelta
from logging import handlers
from platform import platform

# Type Hinting
from typing import Any, Optional, Tuple

import numpy as np
from pandas import (
    DatetimeIndex,
    Index,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    to_datetime,
)
from pandas.tseries.frequencies import to_offset
from scipy import interpolate

from pastas.typing import ArrayLike
from pastas.typing import Model as ModelType
from pastas.typing import TimestampType

logger = logging.getLogger(__name__)


def frequency_is_supported(freq: str) -> str:
    """Method to determine if a frequency is supported for a Pastas model.

    Parameters
    ----------
    freq: str

    Returns
    -------
    freq: str
        String with the simulation frequency

    Notes
    -----
    Possible frequency-offsets are listed in:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    The frequency can be a multiple of these offsets, like '7D'. Because of the
    use in convolution, only frequencies with an equidistant offset are
    allowed. This means monthly ('M'), yearly ('Y') or even weekly ('W')
    frequencies are not allowed. Use '7D' for a weekly simulation.

    D   calendar day frequency
    H   hourly frequency
    T, min      minutely frequency
    S   secondly frequency
    L, ms       milliseconds
    U, us       microseconds
    N   nanoseconds

    TODO: Rename to get_frequency_string and change Returns-documentation
    """
    offset = to_offset(freq)
    if not hasattr(offset, "delta"):
        msg = "Frequency {} not supported.".format(freq)
        logger.error(msg)
        raise KeyError(msg)
    else:
        if offset.n == 1:
            freq = offset.name
        else:
            freq = str(offset.n) + offset.name
    return freq


def _get_dt(freq: str) -> float:
    """Internal method to obtain a timestep in DAYS from a frequency string.

    Parameters
    ----------
    freq: str

    Returns
    -------
    dt: float
        Number of days
    """
    # Get the frequency string and multiplier
    dt = to_offset(freq).delta / Timedelta(1, "D")
    return dt


def _get_time_offset(t: Timestamp, freq: str) -> Timedelta:
    """Internal method to calculate the time offset of a Timestamp.

    Parameters
    ----------
    t: pandas.Timestamp
        Timestamp to calculate the offset from the desired freq for.
    freq: str
        String with the desired frequency.

    Returns
    -------
    offset: pandas.Timedelta
        Timedelta with the offset for the timestamp t.
    """
    if freq is None:
        raise TypeError("frequency is None")

    return t - t.floor(freq)


def get_sample(tindex: Index, ref_tindex: Index) -> Index:
    """Sample the index so that the frequency is not higher than the frequency
    of ref_tindex.

    Parameters
    ----------
    tindex: pandas.index
        Pandas index object
    ref_tindex: pandas.index
        Pandas index object

    Returns
    -------
    series: pandas.index

    Notes
    -----
    Find the index closest to the ref_tindex, and then return a selection
    of the index.
    """
    if len(tindex) == 1:
        return tindex
    else:
        f = interpolate.interp1d(
            tindex.asi8,
            np.arange(0, tindex.size),
            kind="nearest",
            bounds_error=False,
            fill_value="extrapolate",
        )
        ind = np.unique(f(ref_tindex.asi8).astype(int))
        return tindex[ind]


def timestep_weighted_resample(series0: Series, tindex: Index) -> Series:
    """Resample a timeseries to a new tindex, using an overlapping period
    weighted average.

    The original series and the new tindex do not have to be equidistant. Also,
    the timestep-edges of the new tindex do not have to overlap with the
    original series.

    It is assumed the series consists of measurements that describe an
    intensity at the end of the period for which they apply. Therefore, when
    upsampling, the values are uniformly spread over the new timestep (like
    bfill).

    Compared to the reample methods in Pandas, this method is more accurate for
    non-equidistanct series. It is much slower however.

    Parameters
    ----------
    series0 : pandas.Series
        The original series to be resampled
    tindex : pandas.index
        The index to which to resample the series

    Returns
    -------
    series : pandas.Series
        The resampled series
    """

    # determine some arrays for the input-series
    t0e = np.array(series0.index)
    dt0 = np.diff(t0e)
    dt0 = np.hstack((dt0[0], dt0))
    t0s = t0e - dt0
    v0 = series0.values

    # determine some arrays for the output-series
    t1e = np.array(tindex)
    dt1 = np.diff(t1e)
    dt1 = np.hstack((dt1[0], dt1))
    t1s = t1e - dt1
    v1 = []
    for t1si, t1ei in zip(t1s, t1e):
        # determine which periods within the series are within the new tindex
        mask = (t0e > t1si) & (t0s < t1ei)
        if np.any(mask):
            # cut by the timestep-edges
            ts = t0s[mask]
            te = t0e[mask]
            ts[ts < t1si] = t1si
            te[te > t1ei] = t1ei
            # determine timestep
            dt = (te - ts).astype(float)
            # determine timestep-weighted value
            v1.append(np.sum(dt * v0[mask]) / np.sum(dt))
    # replace all values in the series
    series = Series(v1, index=tindex)
    return series


def timestep_weighted_resample_fast(series0: Series, freq: str) -> Series:
    """Resample a time series to a new frequency, using an overlapping period
    weighted average.

    The original series does not have to be equidistant.

    It is assumed the series consists of measurements that describe an
    intensity at the end of the period for which they apply. Therefore, when
    upsampling, the values are uniformly spread over the new timestep (like
    bfill).

    Compared to the resample methods in Pandas, this method is more accurate
    for non-equidistant series. It is slower than Pandas (but faster then the
    original timestep_weighted_resample).

    Parameters
    ----------
    series0 : pandas.Series
        original series to be resampled
    freq : str
        a Pandas frequency string

    Returns
    -------
    series : pandas.Series
        resampled series
    """
    series = series0.copy()

    # first mutiply by the timestep in the unit of freq
    dt = np.diff(series0.index) / Timedelta(1, freq)
    series[1:] = series[1:] * dt

    # get a new index
    index = date_range(series.index[0].floor(freq), series.index[-1], freq=freq)

    # calculate the cumulative sum
    series = series.cumsum()

    # add NaNs at none-existing values in series at index
    series = series.combine_first(Series(np.NaN, index=index))

    # interpolate these NaN's, only keep values at index
    series = series.interpolate("time")[index]

    # calculate the diff again (inverse of cumsum)
    series[1:] = series.diff()[1:]

    # drop nan's at the beginning
    series = series[series.first_valid_index() :]

    return series


def get_equidistant_series(
    series: Series, freq: str, minimize_data_loss: bool = False
) -> Series:
    """Get equidistant timeseries using nearest reindexing.

    This method will shift observations to the nearest equidistant timestep to
    create an equidistant timeseries, if necessary. Each observation is
    guaranteed to only be used once in the equidistant timeseries.

    Parameters
    ----------
    series : pandas.Series
        original (non-equidistant) timeseries
    freq : str
        frequency of the new equidistant timeseries
        (i.e. "H", "D", "7D", etc.)
    minimize_data_loss : bool, optional
        if set to True, method will attempt use any unsampled
        points from original timeseries to fill some remaining
        NaNs in the new equidistant timeseries. Default is False.
        This only happens in rare cases.

    Returns
    -------
    s : pandas.Series
        equidistant timeseries

    Notes
    -----
    This method creates an equidistant timeseries with specified freq
    using nearest sampling (meaning observations can be shifted in time),
    with additional filling logic that ensures each original measurement
    is only included once in the new timeseries. Values are filled as close
    as possible to their original timestamp in the new equidistant timeseries.
    """

    # build new equidistant index
    idx = date_range(
        series.index[0].floor(freq), series.index[-1].ceil(freq), freq=freq
    )

    # get linear interpolated index from original series
    fl = interpolate.interp1d(
        series.index.asi8,
        np.arange(0, series.index.size),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    ind_linear = fl(idx.asi8)

    # get nearest index from original series
    f = interpolate.interp1d(
        series.index.asi8,
        np.arange(0, series.index.size),
        kind="nearest",
        bounds_error=False,
        fill_value="extrapolate",
    )
    ind = f(idx.asi8).astype(int)

    # create a new equidistant series
    s = Series(index=idx, data=np.nan)

    # fill in nearest value for each timestamp in equidistant series
    s.loc[idx] = series.values[ind]

    # remove duplicates, each observation can only be used once
    mask = Series(ind).duplicated(keep=False).values
    # mask all duplicates and set to NaN
    s.loc[mask] = np.nan

    # look through duplicates which equidistant timestamp is closest
    # then fill value from original series for closest timestamp
    for i in np.unique(ind[mask]):
        # mask duplicates
        dupe_mask = ind == i
        # get location of first duplicate
        first_dupe = np.nonzero(dupe_mask)[0][0]
        # get index for closest equidistant timestamp
        i_nearest = np.argmin(np.abs(ind_linear - ind)[dupe_mask])
        # fill value
        s.iloc[first_dupe + i_nearest] = series.values[i]

    # This next part is an ugly bit of code to fill up any
    # nans if there are unused values in the original timeseries
    # that lie close enough to our missing datapoint in the new equidisant
    # series.
    if minimize_data_loss:
        # find remaining nans
        nanmask = s.isna()
        if nanmask.sum() > 0:
            # get unused (not sampled) timestamps from original series
            unused = set(range(series.index.size)) - set(ind)
            if len(unused) > 0:
                # dropna: do not consider unused nans
                missing_ts = series.iloc[list(unused)].dropna().index
                # loop through nan timestamps in new series
                for t in s.loc[nanmask].index:
                    # find closest unused value
                    closest = np.argmin(np.abs(missing_ts - t))
                    # check if value is not farther away that freq to avoid
                    # weird behavior
                    if np.abs(missing_ts[closest] - t) <= Timedelta(freq):
                        # fill value
                        s.loc[t] = series.loc[missing_ts[closest]]
    return s


def to_daily_unit(series: Series, method: bool = True) -> Series:
    """Experimental method, use wth caution!

    Recalculate a timeseries of a stress with a non-daily unit (e/g.
    m3/month) to a daily unit (e.g. m3/day). This method just changes
    the values of the timeseries, and does not alter the frequency.
    """
    if method is True or method == "divide":
        dt = series.index.to_series().diff() / Timedelta(1, "D")
        dt[:-1] = dt[1:]
        dt[-1] = np.NaN
        if not ((dt == 1.0) | dt.isna()).all():
            series = series / dt
    return series


def excel2datetime(tindex: DatetimeIndex, freq="D") -> DatetimeIndex:
    """Method to convert excel datetime to pandas timetime objects.

    Parameters
    ----------
    tindex: datetime index
        can be a datetime object or a pandas datetime index.
    freq: str

    Returns
    -------
    datetimes: pandas.datetimeindex
    """
    datetimes = to_datetime("1899-12-30") + Timedelta(tindex, freq)
    return datetimes


def datenum_to_datetime(datenum: float) -> datetime:
    """Convert Matlab datenum into Python datetime.

    Parameters
    ----------
    datenum: float
        date in datenum format

    Returns
    -------
    datetime :
        Datetime object corresponding to datenum.
    """
    days = datenum % 1.0
    return (
        datetime.fromordinal(int(datenum)) + timedelta(days=days) - timedelta(days=366)
    )


def datetime2matlab(tindex: DatetimeIndex) -> ArrayLike:
    mdn = tindex + Timedelta(days=366)
    frac = (tindex - tindex.round("D")).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def get_stress_tmin_tmax(ml: ModelType) -> Tuple[TimestampType, TimestampType]:
    """Get the minimum and maximum time that all of the stresses have data."""
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
        String with the level to log messages to the screen for. Options
        are: "INFO", "WARNING", and "ERROR".

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
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.
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
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.
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
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all submodules
        and packages.
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
        raise Exception error if illegal character is found, default
        is False which only logs a warning

    Returns
    -------
    name: str
        Unchanged name string
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
            msg = f"User-provided name '{name}' contains illegal character."
            msg += f"Please remove '{char}' from name."
            if raise_error:
                raise Exception(msg)
            else:
                logger.warning(msg)

    return name
