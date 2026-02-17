"""Utility functions for working with time series."""

import logging

import numpy as np
from packaging.version import parse as parse_version
from pandas import (
    DataFrame,
    DatetimeIndex,
    Index,
    Series,
    Timedelta,
    Timestamp,
    api,
    date_range,
    infer_freq,
)
from pandas import __version__ as pd_version
from pandas.core.resample import Resampler
from pandas.tseries.frequencies import to_offset

from .decorators import njit

logger = logging.getLogger(__name__)


def _offset_to_timedelta(offset: Timestamp) -> Timedelta:
    """Convert pandas offset to Timedelta for pandas 3.0 compatibility.

    Parameters
    ----------
    offset : pandas.tseries.offsets.BaseOffset
        Pandas offset object from to_offset().

    Returns
    -------
    pandas.Timedelta
        Converted timedelta.

    Raises
    ------
    ValueError
        If the offset cannot be converted directly to a Timedelta.
    """
    # For fixed frequency offsets in pandas 3.0+, use the nanos attribute
    # which is available on all offset objects
    if hasattr(offset, "nanos"):
        return Timedelta(offset.nanos, "ns")
    # Fallback: raise to signal that this offset can't be converted
    raise ValueError(f"Cannot directly convert offset {offset} to Timedelta")


def _frequency_is_supported(freq: str) -> str:
    """Method to check if frequency string is supported by Pastas Model.

    Parameters
    ----------
    freq: str

    Returns
    -------
    freq : str
        return input frequency string if it is supported.

    Raises
    ------
    ValueError
        raises ValueError if frequency string is not supported.

    Notes
    -----
    Possible frequency-offsets are listed in:
    http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    The frequency can be a multiple of these offsets, like '7D'. Because of the use
    in convolution, only frequencies with an equidistant offset are allowed. This
    means monthly ('M'), yearly ('A') or even weekly ('W') frequencies are not
    allowed. Use '7D' for a weekly simulation.

    D   calendar day frequency
    H   hourly frequency
    T, min      minutely frequency
    S   secondly frequency
    L, ms       milliseconds
    U, us       microseconds
    N   nanoseconds

    """
    offset = to_offset(freq)
    try:
        _offset_to_timedelta(offset)
    except Exception as e:
        msg = "Frequency %s not supported."
        logger.error(msg, freq)
        logger.debug(e)
        raise ValueError(msg % freq)
    if offset.n == 1:
        freq = offset.name
    else:
        freq = str(offset.n) + offset.name
    return freq


def _get_stress_dt(freq: str) -> float:
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
    Used for comparison to determine if a time series needs to be up or downsampled.

    See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    for the offset_aliases supported by Pandas.
    """
    if freq is None:
        return None
    # Get the frequency string and multiplier
    offset = to_offset(freq)
    try:
        dt = _offset_to_timedelta(offset) / Timedelta(1, "D")
    except Exception as e:
        logging.debug(e)
        num = offset.n
        freq = offset._prefix
        if freq in ["A", "Y", "AS", "YS", "YE", "BA", "BY", "BAS", "BYS"]:
            # year
            dt = num * 365
        elif freq in ["BQ", "BQS", "Q", "QS"]:
            # quarter
            dt = num * 90
        elif freq in ["BM", "BMS", "CBM", "CBMS", "M", "MS", "ME"]:
            # month
            dt = num * 30
        elif freq in ["SM", "SMS"]:
            # semi-month
            dt = num * 15
        elif freq in ["W"]:
            # week
            dt = num * 7
        elif freq in ["B", "C"]:
            # day
            dt = num
        elif freq in ["BH", "CBH"]:
            # hour
            dt = num * 1.0 / 24.0
        else:
            raise (ValueError("freq of {} not supported".format(freq)))

    # Check if dt can be an integer, if so convert to int
    if not isinstance(dt, int):
        if dt.is_integer():
            dt = int(dt)

    return dt


def _get_dt(freq: str) -> float:
    """Internal method to obtain a timestep in DAYS from a frequency string.

    Parameters
    ----------
    freq: str

    Returns
    -------
    dt: float
        Number of days.
    """
    # Get the frequency string and multiplier
    offset = to_offset(freq)
    if parse_version(pd_version) >= parse_version("3.0.0"):
        dt = _offset_to_timedelta(offset) / Timedelta(1, "D")
    else:
        # Fallback for non-fixed offsets: re-run _get_stress_dt logic
        dt = _get_stress_dt(freq)
    return dt


def _get_time_offset(t: Timestamp | DatetimeIndex, freq: str) -> Timedelta:
    """Internal method to calculate the time offset of a Timestamp.

    Parameters
    ----------
    t: pandas.Timestamp or pandas.DatetimeIndex
        Timestamp to calculate the offset from the desired freq for.
    freq: str
        String with the desired frequency.

    Returns
    -------
    offset: pandas.Timedelta
        Timedelta with the offset for the timestamp(s) t.
    """
    if freq is None:
        raise TypeError("frequency is None")

    return t - t.floor(freq)


def _infer_fixed_freq(tindex: Index) -> str:
    """Internal method to get the frequency string.

    This methods avoids returning anchored offsets, e.g.
    'W-TUE' will return 7D.

    Parameters
    ----------
    tindex : Index
        DateTimeIndex

    Returns
    -------
    str
        frequency string
    """
    if tindex.empty:
        return None
    freq = infer_freq(tindex)
    if freq is None:
        return freq

    offset = to_offset(freq)
    if to_offset(offset.rule_code).n == 1:
        dt = _get_stress_dt(freq)
        return f"{dt}D"

    return freq


def _get_sim_index(tmin: Timestamp, tmax: Timestamp, freq: str, time_offset: Timestamp):
    """Internal method to determine the simulation index

    Parameters
    ----------
    tmin : pandas.Timestamp
        Timestamp of the end date for the simulation period.
    tmax : pandas.Timestamp
        Timestamp of the start date for the simulation period.
    freq : str
        String representing the desired frequency of the time series. Must be one
        of the following: (D, h, m, s, ms, us, ns) or a multiple of that e.g. "7D".
    time_offset : pandas.Timedelta
        Timedelta with the offset for the timestamp t.

    Returns
    -------
    sim_index: pandas.DatetimeIndex
        Pandas DatetimeIndex instance with the datetimes values for which the
        model is simulated.

    """
    tmin = tmin.floor(freq) + time_offset
    sim_index = date_range(tmin, tmax, freq=freq)
    return sim_index


def get_sample(tindex: DatetimeIndex, ref_tindex: DatetimeIndex) -> DatetimeIndex:
    """Sample the index of a pandas Series or DataFrame so that the frequency is not
    higher than the frequency of ref_tindex.

    Parameters
    ----------
    tindex : pandas.DatetimeIndex
        The original Index, consisting of pandas Timestamps.
    ref_tindex : pandas.DatetimeIndex
        A reference Index consisting of pandas Timestamps.

    Returns
    -------
    pandas.DatetimeIndex
        The sampled index, consisting of a subset of the original index tindex. The
        values in tindex that are closest to ref_index are returned.

    Notes
    -----
    Find the index closest to the ref_tindex, and then return a selection
    of the index.
    """
    if len(tindex) == 1:
        return tindex

    # Sort for nearest matching
    tindex = tindex.sort_values()

    indexer = tindex.get_indexer(ref_tindex, method="nearest")

    # Drop invalid matches
    indexer = indexer[indexer >= 0]

    return tindex[np.unique(indexer)]


def get_sample_for_freq(
    s: Series | DataFrame,
    freq: str,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
):
    """Sample a pandas Series or DataFrame so that the frequency is not higher than a
    supplied frequency.

    Parameters
    ----------
    s : pandas.Series or pandas.DataFrame
        The original Series or DataFrame to be sampled.
    freq : str
        A frequency string accepted by `pandas.date_range()`.
    tmin : pandas.Timestamp or str, optional
        The start date of the sampled series. If None, the tmin is set to the first
        index of s. The default is None.
    tmax : pandas.Timestamp or str, optional
        The end date of the sampled series. If None, the tmax is set to the last
        index of s. The default is None.

    Returns
    -------
    pandas.Series
        The sampled series, consisting of a subset of the original series.

    """
    if tmin is None:
        tmin = s.index.min()
    if tmax is None:
        tmax = s.index.max()
    ref_tindex = date_range(tmin, tmax, freq=freq)
    return s.loc[get_sample(s.index, ref_tindex)]


def timestep_weighted_resample(s: Series, index: Index, fast: bool = False) -> Series:
    """Resample a time series to a new time index, using an overlapping period
    weighted average.

    The original series and the new index do not have to be equidistant. Also, the
    timestep-edges of the new index do not have to overlap with the original series.

    It is assumed the series consists of measurements that describe a flux intensity
    that for each record starts at the previous index and ends at its own index. So the
    index of the series describes the end of the period for a given measurement.

    When upsampling, the values are uniformly spread over the new timestep (like bfill).
    When downsampling, the values are aggregated to the new index. When the start and
    end of the new index do not overlap with the series (eg: resampling pecipitation
    from 9:00 to 0:00), new values are calculated by combining original measurements.

    Compared to the resample methods in Pandas, this method is more accurate for
    non-equidistant series.

    Parameters
    ----------
    s : pandas.Series
        The original series to be resampled
    index : pandas.Index
        The index to which to resample the series
    fast : bool, optional
        use fast implementation, default is False

    Returns
    -------
    s_new : pandas.Series
        The resampled series
    """
    if isinstance(s, DataFrame):
        if len(s.columns) == 1:
            s = s.iloc[:, 0]
        elif len(s.columns) > 1:
            # helpful specific message for multi-column DataFrames
            msg = "DataFrame with multiple columns. Please select one."
            logger.error(msg)
            raise ValueError(msg)

    dt = _get_dt_array(s.index)

    if fast:
        if s.isna().any():
            raise Exception("s cannot contain NaN values when fast=True")
        if not api.types.is_float_dtype(s):
            raise Exception("s must be of dtype float")

        # first multiply by the timestep
        s_new = s * dt

        # calculate the cumulative sum
        s_new = s_new.cumsum()

        # add NaNs at non-existing values in series at index
        s_new = s_new.combine_first(Series(np.nan, index))

        # interpolate these NaN's, only keep values at index
        s_new = s_new.interpolate("time")[index]

        # calculate the diff again (inverse of cumsum)
        s_new = s_new.diff()

        # divide by the timestep again
        s_new = s_new / _get_dt_array(s_new.index)

        # set values after the end of the original series to NaN
        s_new[s_new.index > s.index[-1]] = np.nan
    else:
        t_e = s.index.asi8
        t_s = t_e - dt
        v = s.values
        t_new = index.asi8
        v_new = _ts_resample_slow(t_s, t_e, v, t_new)
        s_new = Series(v_new, index)

    return s_new


def _get_dt_array(index):
    dt = np.diff(index.asi8)
    # assume the first value has an equal timestep as the second value
    dt = np.hstack((dt[0], dt))
    return dt


@njit
def _ts_resample_slow(t_s, t_e, v, t_new):
    v_new = np.full(t_new.shape, np.nan)
    for i in range(1, len(t_new)):
        t_s_new = t_new[i - 1]
        t_e_new = t_new[i]
        if t_s_new < t_s[0] or t_e_new > t_e[-1]:
            continue
        # determine which periods within the series are within the new tindex
        mask = (t_e > t_s_new) & (t_s < t_e_new)
        if not np.any(mask):
            continue
        ts = t_s[mask]
        te = t_e[mask]
        ts[ts < t_s_new] = t_s_new
        te[te > t_e_new] = t_e_new
        # determine timestep
        dt = te - ts
        # determine timestep-weighted value
        v_new[i] = np.sum(dt * v[mask]) / np.sum(dt)
    return v_new


def _get_nearest_offset_to_freq(tindex: DatetimeIndex, freq: str) -> Timedelta:
    """Internal method to get the nearest offset to a frequency string.

    Parameters
    ----------
    tindex : pandas.DatetimeIndex
        The index to calculate the offset from.
    freq : str
        The frequency string to calculate the offset for.

    Returns
    -------
    pandas.Timedelta
        The nearest offset to the frequency string.
    """
    offsets = _get_time_offset(t=tindex, freq=freq).value_counts()
    t_offset = offsets.idxmax() if len(offsets) > 0 else Timedelta(0)
    return t_offset


def get_equidistant_series_nearest(
    series: Series, freq: str, minimize_data_loss: bool = False
) -> Series:

    if len(series) == 0:
        return series

    # Must be sorted for nearest matching
    series = series.sort_index()

    # Build equidistant index
    t_offset = _get_nearest_offset_to_freq(series.index, freq)

    idx = date_range(
        series.index[0].floor(freq) + t_offset,
        series.index[-1].ceil(freq) + t_offset,
        freq=freq,
    )

    # Nearest matching
    ind = series.index.get_indexer(idx, method="nearest")

    # Remove out-of-range matches
    valid = ind >= 0
    ind = ind[valid]

    s = Series(index=idx, dtype=float)

    # Initial fill
    s.iloc[valid] = series.values[ind]

    # ---- Duplicate resolution (each original value used once) ----

    dup_mask = Series(ind).duplicated(keep=False).values
    s.iloc[np.where(valid)[0][dup_mask]] = np.nan

    for i in np.unique(ind[dup_mask]):
        dupe_positions = np.where(ind == i)[0]

        # choose closest in actual time distance
        distances = np.abs(idx[dupe_positions].view("int64") - series.index[i].value)

        best = dupe_positions[np.argmin(distances)]
        s.iloc[best] = series.iloc[i]

    # ---- Minimize data loss ----
    if minimize_data_loss:
        nanmask = s.isna()

        if nanmask.any():
            used = set(ind)
            unused = list(set(range(len(series))) - used)

            if unused:
                unused_idx = series.index[unused]
                unused_vals = series.iloc[unused]

                for t in s.index[nanmask]:
                    distances = np.abs(unused_idx.view("int64") - t.value)
                    closest = np.argmin(distances)

                    if distances[closest] <= Timedelta(freq).value:
                        s.loc[t] = unused_vals.iloc[closest]

    return s


def pandas_equidistant_sample(series: Series, freq: str) -> Series:
    """Create equidistant time series creating a new DateTimeIndex with pandas.reindex.

    Note: function attempts to pick an offset relative to freq such that the maximum
    number of observations is included in the sample.

    Parameters
    ----------
    series : pandas Series
        time series
    freq : str
        frequency str

    Returns
    -------
    Series
        equidistant time series with frequency 'freq'
    """
    series = series.copy()
    # find most common offset relative to freq
    offsets = _get_time_offset(series.index, freq).value_counts()
    t_offset = offsets.idxmax() if len(offsets) > 0 else Timedelta(0)
    # use t_offset to pick time that will keep the most data from the original series.
    new_idx = date_range(
        series.index[0].floor(freq) + t_offset,
        series.index[-1].floor(freq) + t_offset,
        freq=freq,
    )
    return series.reindex(new_idx)


def pandas_equidistant_nearest(
    series: Series, freq: str, tolerance: str | None = None
) -> Series:
    """Create equidistant time series using pandas.reindex with method='nearest'.

    Note: this method will shift observations in time and can introduce duplicate
    observations.

    Parameters
    ----------
    series : str
        time series.
    freq : str
        frequency string.
    tolerance : str, optional
        frequency type string (e.g. '7D') specifying maximum distance between original
        and new labels for inexact matches.

    Returns
    -------
    Series
        equidistant time series with frequency 'freq'
    """
    series = series.copy()
    # Create equidistant time index
    idx = date_range(
        series.index[0].floor(freq), series.index[-1].ceil(freq), freq=freq
    )
    # reindex with nearest and optional tolerance
    spandas = series.reindex(idx, method="nearest", tolerance=tolerance)
    return spandas


def pandas_equidistant_asfreq(series: Series, freq: str) -> Series:
    """Create equidistant time series by rounding timestamps and dropping duplicates.

    Note: this method rounds all timestamps down to the nearest "freq" then drops
    duplicates by keeping the first entry. This means observations are shifted in time.

    Parameters
    ----------
    series : pandas Series
        time series.
    freq : str
        frequency string.

    Returns
    -------
    Series
        equidistant time series with frequency 'freq'.
    """
    series = series.copy()
    # round to the nearest freq
    series.index = series.index.floor(freq)
    # keep first entry for each duplicate
    spandas = (
        series.reset_index()
        .drop_duplicates(subset="index", keep="first", inplace=False)
        .set_index("index")
        .asfreq(freq)
        .squeeze()
    )
    return spandas


def resample(
    series: Series, freq: str, closed: str = "right", label: str = "right", **kwargs
) -> Resampler:
    """Resample time-series data.

    Convenience method for frequency conversion and resampling of time series.
    This function is a wrapper around Pandas' resample function with some
    logical Pastas defaults. In Pastas, we assume the timestamp is at the end
    of the period that belongs to each measurement. For more information on
    this read the example notebook on preprocessing time series.

    Parameters
    ----------
    series : pandas Series
        Time series. The index must be a datetime-like index
        (`DatetimeIndex`, `PeriodIndex`, or `TimedeltaIndex`).
    freq : str
        Frequency string.
    closed: str, default 'right'
        Which side/end of bin interval is closed.
    label: str, default 'right'
        Which bin edge label to label bucket with.
    **kwargs: dict

    Returns
    -------
    Resampler
        pandas Resampler object which can be manipulated using methods such as:
        '.interpolate()', '.mean()', '.max()' etc. For all options see:
        https://pandas.pydata.org/docs/reference/resampling.html

    """

    return series.resample(freq, closed=closed, label=label, **kwargs)
