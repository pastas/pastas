import datetime
import pandas as pd


def get_dt(freq):
    # method to calculate the timestep in days from the frequency string freq
    options = {'MS': 30,  # monthly frequency (month-start), used just for comparison
               'M': 30,  # monthly frequency (month-end), used just for comparison
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


def get_time_offset(t, freq):
    # method to calculate the time offset between a TimeStamp t and a default Series with a frequency of freq
    if isinstance(t, pd.Series):
        # Take the first timestep. The rest of index has the same offset,
        # as the frequency is constant.
        t = t.index[0]

    # define the function blocks
    def calc_week_offset(t):
        return datetime.timedelta(days=t.weekday(), hours=t.hour,
                                  minutes=t.minute, seconds=t.second)

    def calc_day_offset(t):
        return datetime.timedelta(hours=t.hour, minutes=t.minute,
                                  seconds=t.second)

    def calc_hour_offset(t):
        return datetime.timedelta(minutes=t.minute, seconds=t.second)

    def calc_minute_offset(t):
        return datetime.timedelta(seconds=t.second)

    def calc_second_offset(t):
        return datetime.timedelta(microseconds=t.microsecond)

    def calc_millisecond_offset(t):
        # t has no millisecond attribute, so use microsecond and use the remainder after division by 1000
        return datetime.timedelta(microseconds=t.microsecond % 1000.0)

    # map the inputs to the function blocks
    # see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#timeseries-offset-aliases
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

