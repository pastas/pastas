"""This module contains utility functions for working with Pastas models.

"""

import logging
from datetime import datetime, timedelta
from logging import handlers

import numpy as np
from pandas import Series, to_datetime, Timedelta, Timestamp, date_range
from pandas.tseries.frequencies import to_offset
from scipy import interpolate

logger = logging.getLogger(__name__)


def frequency_is_supported(freq):
    """Method to determine if a frequency is supported for a Pastas model.

    Parameters
    ----------
    freq: str

    Returns
    -------
    freq
        String with the simulation frequency

    Notes
    -----
    Possible frequency-offsets are listed in:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    The frequency can be a multiple of these offsets, like '7D'. Because of the
    use in convolution, only frequencies with an equidistant offset are
    allowed. This means monthly ('M'), yearly ('Y') or even weekly ('W')
    frequencies are not allowed. Use '7D' for a weekly simulation.

    D	calendar day frequency
    H	hourly frequency
    T, min	minutely frequency
    S	secondly frequency
    L, ms	milliseconds
    U, us	microseconds
    N	nanoseconds

    TODO: Rename to get_frequency_string and change Returns-documentation

    """
    offset = to_offset(freq)
    if not hasattr(offset, 'delta'):
        msg = "Frequency {} not supported.".format(freq)
        logger.error(msg)
        raise KeyError(msg)
    else:
        if offset.n == 1:
            freq = offset.name
        else:
            freq = str(offset.n) + offset.name
    return freq


def _get_stress_dt(freq):
    """Internal method to obtain a timestep in days from a frequency string.

    Parameters
    ----------
    freq: str

    Returns
    -------
    dt: float
        Approximate timestep in number of days.

    Notes
    -----
    Used for comparison to determine if a time series needs to be up or
    downsampled.

    See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    for the offset_aliases supported by Pandas.

    """
    # Get the frequency string and multiplier
    offset = to_offset(freq)
    if hasattr(offset, 'delta'):
        dt = offset.delta / Timedelta(1, "D")
    else:
        num = offset.n
        freq = offset.name
        if freq in ['A', 'Y', 'AS', 'YS', 'BA', 'BY', 'BAS', 'BYS']:
            # year
            dt = num * 365
        elif freq in ['BQ', 'BQS', 'Q', 'QS']:
            # quarter
            dt = num * 90
        elif freq in ['BM', 'BMS', 'CBM', 'CBMS', 'M', 'MS']:
            # month
            dt = num * 30
        elif freq in ['SM', 'SMS']:
            # semi-month
            dt = num * 15
        elif freq in ['W']:
            # week
            dt = num * 7
        elif freq in ['B', 'C']:
            # day
            dt = num
        elif freq in ['BH', 'CBH']:
            # hour
            dt = num * 1 / 24
        else:
            raise (ValueError('freq of {} not supported'.format(freq)))

    return dt


def _get_dt(freq):
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


def _get_time_offset(t, freq):
    """Internal method to calculate the time offset of a TimeStamp.

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


def get_sample(tindex, ref_tindex):
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
        f = interpolate.interp1d(tindex.asi8, np.arange(0, tindex.size),
                                 kind='nearest', bounds_error=False,
                                 fill_value='extrapolate')
        ind = np.unique(f(ref_tindex.asi8).astype(int))
        return tindex[ind]


def timestep_weighted_resample(series0, tindex):
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


def timestep_weighted_resample_fast(series0, freq):
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
    index = date_range(series.index[0].floor(freq), series.index[-1],
                       freq=freq)

    # calculate the cumulative sum
    series = series.cumsum()

    # add NaNs at none-existing values in series at index
    series = series.combine_first(Series(np.NaN, index=index))

    # interpolate these NaN's, only keep values at index
    series = series.interpolate('time')[index]

    # calculate the diff again (inverse of cumsum)
    series[1:] = series.diff()[1:]

    # drop nan's at the beginning
    series = series[series.first_valid_index():]

    return series


def to_daily_unit(series, method=True):
    """Experimental method, use wth caution!

    Recalculate a timeseries of a stress with a non-daily unit (e/g.
    m3/month) to a daily unit (e.g. m3/day). This method just changes the
    values of the timeseries, and does not alter the frequency.

    """
    if method is True or method == "divide":
        dt = series.index.to_series().diff() / Timedelta(1, 'D')
        dt[:-1] = dt[1:]
        dt[-1] = np.NaN
        if not ((dt == 1.0) | dt.isna()).all():
            series = series / dt
    return series


def excel2datetime(tindex, freq="D"):
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
    datetimes = to_datetime('1899-12-30') + Timedelta(tindex, freq)
    return datetimes


def datenum_to_datetime(datenum):
    """
    Convert Matlab datenum into Python datetime.
    Parameters
    ----------
    datenum: float
        date in datenum format

    Returns
    -------
    datetime :
        Datetime object corresponding to datenum.
    """
    days = datenum % 1.
    return datetime.fromordinal(int(datenum)) \
           + timedelta(days=days) - timedelta(days=366)


def datetime2matlab(tindex):
    mdn = tindex + Timedelta(days=366)
    frac = (tindex - tindex.round("D")).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac


def get_stress_tmin_tmax(ml):
    """Get the minimum and maximum time that all of the stresses have data"""
    from .model import Model
    from .project import Project
    tmin = Timestamp.min
    tmax = Timestamp.max
    if isinstance(ml, Model):
        for sm in ml.stressmodels:
            for st in ml.stressmodels[sm].stress:
                tmin = max((tmin, st.series_original.index.min()))
                tmax = min((tmax, st.series_original.index.max()))
    elif isinstance(ml, Project):
        for st in ml.stresses['series']:
            tmin = max((tmin, st.series_original.index.min()))
            tmax = min((tmax, st.series_original.index.max()))
    else:
        raise (TypeError('Unknown type {}'.format(type(ml))))
    return tmin, tmax


def initialize_logger(logger=None, level=logging.INFO):
    """Internal method to create a logger instance to log program output.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('pastas')
    logger.setLevel(level)
    remove_file_handlers(logger)
    set_console_handler(logger)
    # add_file_handlers(logger)


def set_console_handler(logger=None, level=logging.INFO,
                        fmt="%(levelname)s: %(message)s"):
    """Method to add a console handler to the logger of Pastas.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('pastas')
    remove_console_handler(logger)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter(fmt=fmt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def set_log_level(level):
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


def remove_console_handler(logger=None):
    """Method to remove the console handler to the logger of Pastas.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('pastas')
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)


def add_file_handlers(logger=None, filenames=('info.log', 'errors.log'),
                      levels=(logging.INFO, logging.ERROR), maxBytes=10485760,
                      backupCount=20, encoding='utf8',
                      fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                      datefmt='%y-%m-%d %H:%M'):
    """Method to add file handlers in the logger of Pastas

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('pastas')
    # create formatter
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # create file handlers, set the level & formatter, and add it to the logger
    for filename, level in zip(filenames, levels):
        fh = handlers.RotatingFileHandler(filename, maxBytes=maxBytes,
                                          backupCount=backupCount,
                                          encoding=encoding)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def remove_file_handlers(logger=None):
    """Method to remove any file handlers in the logger of Pastas.

    Parameters
    -------
    logger : logging.Logger
        A Logger-instance. Use ps.logger to initialise the Logging instance
        that handles all logging throughout pastas,  including all sub modules
        and packages.

    """
    if logger is None:
        logger = logging.getLogger('pastas')
    for handler in logger.handlers:
        if isinstance(handler, handlers.RotatingFileHandler):
            logger.removeHandler(handler)


def validate_name(name):
    """Method to check user-provided names and log a warning if wrong.

    Parameters
    ----------
    name: str
        String with the name to check for 'illegal' characters.

    Returns
    -------
    name: str
        Unchanged name string

    Notes
    -----
    Forbidden characters are: "/", "\", " ".

    """
    name = str(name)  # Make sure it is a string

    for char in ["\\", "/", " "]:
        if char in name:
            msg = "User-provided name '{}' contains illegal character " \
                  "{}".format(name, char)
            logger.warning(msg)

    return name


def show_versions(lmfit=False, numba=False):
    """Method to print the version of dependencies.

    Parameters
    ----------
    lmfit: bool, optional
        Print the version of lmfit. Needs to be installed.
    numba: bool, optional
        Print the version of numba. Needs to be installed.

    """
    from pastas import __version__ as ps_version
    from pandas import __version__ as pd_version
    from numpy import __version__ as np_version
    from scipy import __version__ as sc_version
    from matplotlib import __version__ as mpl_version
    from sys import version as os_version

    msg = (
        f"Python version: {os_version}\n"
        f"Numpy version: {np_version}\n"
        f"Scipy version: {sc_version}\n"
        f"Pandas version: {pd_version}\n"
        f"Pastas version: {ps_version}\n"
        f"Matplotlib version: {mpl_version}"
    )

    if lmfit:
        from lmfit import __version__ as lm_version
        msg = msg + f"\nlmfit version: {lm_version}"
    if numba:
        from numba import __version__ as nb_version
        msg = msg + f"\nnumba version: {nb_version}"

    return print(msg)
