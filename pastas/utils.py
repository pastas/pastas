import logging

import numpy as np
from pandas import Series, to_datetime, Timedelta, Timestamp, to_timedelta
from scipy import interpolate

logger = logging.getLogger(__name__)

_unit_map = {
    'Y': 'Y',
    'y': 'Y',
    'W': 'W',
    'w': 'W',
    'D': 'D',
    'd': 'D',
    'days': 'D',
    'Days': 'D',
    'day': 'D',
    'Day': 'D',
    'M': 'M',
    'H': 'h',
    'h': 'h',
    'm': 'm',
    'min': 'm',
    'T': 'm',
    't': 'm',
    'S': 's',
    's': 's',
    'L': 'ms',
    'MS': 'ms',
    'ms': 'ms',
    'US': 'us',
    'us': 'us',
    'NS': 'ns',
    'ns': 'ns',
}


def frequency_is_supported(freq):
    num, freq = get_freqstr(freq)
    return freq in _unit_map.keys()


def get_dt(freq):
    """Method to obtain a timestep in days from a frequency string.

    Parameters
    ----------
    freq: str

    Returns
    -------
    dt: float

    """
    # Get the frequency string and multiplier
    num, freq = get_freqstr(freq)

    if freq == "W":  # Deal with weeks.
        num = num * 7
        freq = "D"

    dt_str = str(num) + freq
    dt = to_timedelta(dt_str) / Timedelta(1, "D")
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
    # Get the frequency string and multiplier
    num, freq = get_freqstr(freq)

    if freq in ["W"]:
        offset = Timedelta(days=t.weekday(), hours=t.hour,
                           minutes=t.minute, seconds=t.second)
    else:
        offset = t - t.round(freq)

    return offset


def get_freqstr(freqstr):
    """Method to untangle the frequency string.

    Parameters
    ----------
    freqstr: str
        string with the frequency as defined by the pandas package,
        possibly containing a numerical value.

    Returns
    -------
    num: int
        integer by which to multiply the frequency. 1 is returned if no
        num is present in the string that has been provided.
    freq: str
        String with the frequency as defined by the pandas package.

    """
    # remove the day from the week
    freqstr = freqstr.split("-", 1)[0]

    # Find a number by which the frequency is multiplied
    num = ""
    freq = ""
    for s in freqstr:
        if s.isdigit():
            num = num.__add__(s)
        else:
            freq = freq.__add__(s)
    if num:
        num = int(num)
    else:
        num = 1

    if freq not in _unit_map.keys():
        logger.error("Frequency %s not supported." % freq)
    else:
        freq = _unit_map[freq]

    return num, freq


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
        f = interpolate.interp1d(tindex.asi8,
                                 np.arange(0, tindex.size),
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
        if any(mask):
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
