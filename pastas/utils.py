from logging import getLogger

import numpy as np
from pandas import Series, to_datetime, Timedelta, Timestamp, to_timedelta
from pandas.tseries.frequencies import to_offset
from scipy import interpolate

logger = getLogger(__name__)


def frequency_is_supported(freq):
    """Method to determine if a frequency is supported for a  pastas-model.
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

    Parameters
    ----------
    freq: str

    Returns
    -------
    boolean
        True when frequency can be used as a simulation frequency
    """

    offset = to_offset(freq)
    if not hasattr(offset, 'delta'):
        logger.error("Frequency %s not supported." % freq)
    else:
        if offset.n == 1:
            freq = offset.name
        else:
            freq = str(offset.n) + offset.name
    return freq


def get_stress_dt(freq):
    """Internal method to obtain a timestep in days from a frequency string
    derived by Pandas Infer method or supplied by the user as a TimeSeries
    settings.

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


def get_dt(freq):
    """Method to obtain a timestep in DAYS from a frequency string.

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


def get_time_offset(t, freq):
    """ method to calculate the time offset between a TimeStamp t and a
    default Series with a frequency of freq

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


def timestep_weighted_resample(series, tindex):
    """resample a timeseries to a new tindex, using an overlapping-timestep
    weighted average the new tindex does not have to be equidistant also,
    the timestep-edges of the new tindex do not have to overlap with the
    original series it is assumed the series consists of measurements that
    describe an intensity at the end of the period for which they hold
    therefore when upsampling, the values are uniformally spread over the
    new timestep (like bfill) this method unfortunately is slower than the
    pandas-reample methods.

    Parameters
    ----------
    series
    tindex

    Returns
    -------

    TODO Make faster, document and test.

    """

    # determine some arrays for the input-series
    t0e = series.index.get_values()
    dt0 = np.diff(t0e)
    dt0 = np.hstack((dt0[0], dt0))
    t0s = t0e - dt0
    v0 = series.values

    # determine some arrays for the output-series
    t1e = tindex.get_values()
    dt1 = np.diff(t1e)
    dt1 = np.hstack((dt1[0], dt1))
    t1s = t1e - dt1
    v1 = np.empty(t1e.shape)
    v1[:] = np.nan
    for i in range(len(v1)):
        # determine which periods within the series are within the new tindex
        mask = (t0e > t1s[i]) & (t0s < t1e[i])
        if np.any(mask):
            # cut by the timestep-edges
            ts = t0s[mask]
            te = t0e[mask]
            ts[ts < t1s[i]] = t1s[i]
            te[te > t1e[i]] = t1e[i]
            # determine timestep
            dt = (te - ts).astype(float)
            # determine timestep-weighted value
            v1[i] = np.sum(dt * v0[mask]) / np.sum(dt)
    # replace all values in the series
    series = Series(v1, index=tindex)
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
    datetimes = to_datetime('1899-12-30') + to_timedelta(tindex, freq)
    return datetimes


def matlab2datetime(tindex):
    """ Transform a matlab time to a datetime, rounded to seconds

    """
    day = Timestamp.fromordinal(int(tindex))
    dayfrac = Timedelta(days=float(tindex) % 1) - Timedelta(days=366)
    return day + dayfrac


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
