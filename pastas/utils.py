import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import interpolate


def get_dt(freq):
    """Method to obtain a timestep from a frequency string.

    Parameters
    ----------
    freq: str

    Returns
    -------
    dt: float

    """
    # method to calculate the timestep in days from the frequency string freq
    options = {'MS': 30,  # monthly frequency (month-start), used just for
               # comparison
               'M': 30,  # monthly frequency (month-end), used just for
               # comparison
               'W': 7,  # weekly frequency
               'D': 1,  # calendar day frequency
               'H': 1 / 24,  # hourly frequency
               'T': 1 / 24 / 60,  # minutely frequency
               'min': 1 / 24 / 60,  # minutely frequency
               'S': 1 / 24 / 3600,  # secondly frequency
               'L': 1 / 24 / 3600000,  # milliseconds
               'ms': 1 / 24 / 3600000,  # milliseconds
               }
    # Get the frequency string and multiplier
    num, freq = get_freqstr(freq)
    dt = num * options[freq]
    return dt

def frequency_is_supported(freq):
    num, freq = get_freqstr(freq)
    return freq in ['W','D','H','T','min','S','L','ms']

def get_time_offset(t, freq):
    # method to calculate the time offset between a TimeStamp t and a default Series with a frequency of freq
    if isinstance(t, pd.Series):
        # Take the first timestep. The rest of index has the same offset,
        # as the frequency is constant.
        t = t.index[0]

    # define the function blocks
    def calc_week_offset(t):
        return pd.Timedelta(days=t.weekday(), hours=t.hour,
                            minutes=t.minute, seconds=t.second)

    def calc_day_offset(t):
        return pd.Timedelta(hours=t.hour, minutes=t.minute,
                            seconds=t.second)

    def calc_hour_offset(t):
        return pd.Timedelta(minutes=t.minute, seconds=t.second)

    def calc_minute_offset(t):
        return pd.Timedelta(seconds=t.second)

    def calc_second_offset(t):
        return pd.Timedelta(microseconds=t.microsecond)

    def calc_millisecond_offset(t):
        # t has no millisecond attribute, so use microsecond and use the remainder after division by 1000
        return pd.Timedelta(microseconds=t.microsecond % 1000.0)

    # map the inputs to the function blocks see
    # http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    options = {'W': calc_week_offset,  # weekly frequency
               'D': calc_day_offset,  # calendar day frequency
               'H': calc_hour_offset,  # hourly frequency
               'T': calc_minute_offset,  # minutely frequency
               'min': calc_minute_offset,  # minutely frequency
               'S': calc_second_offset,  # secondly frequency
               'L': calc_millisecond_offset,  # milliseconds
               'ms': calc_millisecond_offset,  # milliseconds
               }
    # Get the frequency string and multiplier
    num, freq = get_freqstr(freq)
    offset = num * options[freq](t)
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
    num = ''
    freq = ''
    for s in freqstr:
        if s.isdigit():
            num = num.__add__(s)
        else:
            freq = freq.__add__(s)
    if num:
        num = int(num)
    else:
        num = 1

    return num, freq


def timestep_weighted_resample(series, index):
    # resample a timeseries to a new index, using an overlapping-timestep weighted average
    # the new index does not have to be equidistant
    # also, the timestep-edges of the new index do not have to overlap with the original series
    # it is assumed the series consists of measurements that describe an intensity at the end of the period for which they hold
    # therefore when upsampling, the values are uniformally spread over the new timestep (like bfill)
    # this method unfortunately is slower than the pandas-reample methods

    # determine some arrays for the input-series
    t0e = series.index.get_values()
    dt0 = np.diff(t0e)
    dt0 = np.hstack((dt0[0], dt0))
    t0s = t0e - dt0
    v0 = series.values

    # determine some arrays for the output-series
    t1e = index.get_values()
    dt1 = np.diff(t1e)
    dt1 = np.hstack((dt1[0], dt1))
    t1s = t1e - dt1
    v1 = np.empty(t1e.shape)
    v1[:] = np.nan
    for i in range(len(v1)):
        # determine which periods within the series are within the new index
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
    series = pd.Series(v1, index=index)
    return series


def excel2datetime(excel_datenum, freq="D"):
    """Method to convert excel datetime to pandas timetime objects.

    Parameters
    ----------
    excel_datenum: datetime index
        can be a datetime object or a pandas datetime index.
    freq:

    Returns
    -------
    datetimes: pandas.datetimeindex

    """
    datetimes = pd.to_datetime('1899-12-30') + pd.to_timedelta(excel_datenum,
                                                               freq)
    return datetimes

def matlab2datetime(matlab_datenum):
    """
    Transform a matlab time to a datetime, rounded to seconds
    """
    day = datetime.fromordinal(int(matlab_datenum))
    dayfrac = timedelta(days=float(matlab_datenum) % 1) - timedelta(
        days=366)
    return day + dayfrac

def datetime2matlab(dt):
    mdn = dt + timedelta(days = 366)
    frac = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return mdn.toordinal() + frac

def get_sample(index, ref_index):
    """Sample the index so that the frequency is not higher than the frequency
        of tindex.

    Parameters
    ----------
    index: pandas.index
        Pandas index object
    ref_index: pandas.index
        Pandas index object

    Returns
    -------
    series: pandas.index

    Notes
    -----
    Find the index closest to the ref_index, and then return a selection
    of the index.

    """
    f = interpolate.interp1d(index.asi8,
                             np.arange(0, index.size),
                             kind='nearest', bounds_error=False,
                             fill_value='extrapolate')
    ind = np.unique(f(ref_index.asi8).astype(int))
    return index[ind]
    
