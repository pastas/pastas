"""This module contains utility functions for working with time series."""

import logging

# Type Hinting
from typing import Optional

import numpy as np
from pandas import Index, Series, Timedelta, Timestamp, date_range, api
from pandas.tseries.frequencies import to_offset
from scipy import interpolate

from .decorators import njit

logger = logging.getLogger(__name__)


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
    means monthly ('M'), yearly ('Y') or even weekly ('W') frequencies are not
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
    if not hasattr(offset, "delta"):
        msg = "Frequency {} not supported.".format(freq)
        logger.error(msg)
        raise ValueError(msg)
    else:
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

    See http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    for the offset_aliases supported by Pandas.
    """
    # Get the frequency string and multiplier
    offset = to_offset(freq)
    if hasattr(offset, "delta"):
        dt = offset.delta / Timedelta(1, "D")
    else:
        num = offset.n
        freq = offset.name
        if freq in ["A", "Y", "AS", "YS", "BA", "BY", "BAS", "BYS"]:
            # year
            dt = num * 365
        elif freq in ["BQ", "BQS", "Q", "QS"]:
            # quarter
            dt = num * 90
        elif freq in ["BM", "BMS", "CBM", "CBMS", "M", "MS"]:
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
        Pandas index object.
    ref_tindex: pandas.index
        Pandas index object.

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
    index : pandas.index
        The index to which to resample the series
    fast : bool, optional
        use fast implementation, default is False

    Returns
    -------
    s_new : pandas.Series
        The resampled series
    """
    dt = _get_dt_array(s.index)

    if fast:
        if s.isna().any():
            raise Exception("s cannot contain NaN values when fast=True")
        if not api.types.is_float_dtype(s):
            raise Exception("s must be of dtype float")

        # first mutiply by the timestep
        s_new = s * dt

        # calculate the cumulative sum
        s_new = s_new.cumsum()

        # add NaNs at none-existing values in series at index
        s_new = s_new.combine_first(Series(np.NaN, index))

        # interpolate these NaN's, only keep values at index
        s_new = s_new.interpolate("time")[index]

        # calculate the diff again (inverse of cumsum)
        s_new = s_new.diff()

        # devide by the timestep again
        s_new = s_new / _get_dt_array(s_new.index)

        # set values after the end of the original series to NaN
        s_new[s_new.index > s.index[-1]] = np.NaN
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
    v_new = np.full(t_new.shape, np.NaN)
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


def get_equidistant_series_nearest(
    series: Series, freq: str, minimize_data_loss: bool = False
) -> Series:
    """Get equidistant time series using nearest reindexing.

    This method will shift observations to the nearest equidistant timestep to create
    an equidistant time series, if necessary. Each observation is guaranteed to only
    be used once in the equidistant time series.

    Parameters
    ----------
    series : pandas.Series
        original (non-equidistant) time series
    freq : str
        frequency of the new equidistant time series (i.e. "H", "D", "7D", etc.)
    minimize_data_loss : bool, optional
        if set to True, method will attempt use any unsampled points from original
        time series to fill some remaining NaNs in the new equidistant time series.
        Default is False. This only happens in rare cases.

    Returns
    -------
    s : pandas.Series
        equidistant time series

    Notes
    -----
    This method creates an equidistant time series with specified freq using nearest
    sampling (meaning observations can be shifted in time), with additional filling
    logic that ensures each original measurement is only included once in the new
    time series. Values are filled as close as possible to their original timestamp
    in the new equidistant time series.
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
    # nans if there are unused values in the original time series
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
    t_offset = _get_time_offset(series.index, freq).value_counts().idxmax()
    # use t_offset to pick time that will keep the most data from the original series.
    new_idx = date_range(
        series.index[0].floor(freq) + t_offset,
        series.index[-1].floor(freq) + t_offset,
        freq=freq,
    )
    return series.reindex(new_idx)


def pandas_equidistant_nearest(
    series: Series, freq: str, tolerance: Optional[str] = None
) -> Series:
    """Create equidistant time series using pandas.reindex with method='nearest'.

    Note: this method will shift observations in time and can introduce duplicate
    observations.

    Parameters
    ----------
    series : str
        _description_
    freq : str
        _description_
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


def to_daily_unit():
    raise DeprecationWarning("This function is deprecated and will be removed in v1.0")


def timestep_weighted_resample_fast():
    raise DeprecationWarning(
        "This function is deprecated and will be removed in v1.0."
        "Method is merged into `timestep_weighted_resample()`."
    )
