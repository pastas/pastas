import pandas as pd


def check_oseries(oseries, freq):
    """Check the observed time series before running a simulation.

    Parameters
    ----------
    oseries: pd.Series
        Pandas series object containing the observed time series.
    freq: str
        String containing the desired frequency. The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases

    Returns
    -------
    oseries: pd.Series
        Pandas series object check for nan-values and with the required frequency.

    """
    assert isinstance(oseries, pd.Series), 'Expected a Pandas Series object, ' \
                                           'got %s' % type(oseries)
    # make a deep copy to preserve original imported data
    oseries = oseries.copy(deep=True)
    # Deal with frequency of the time series
    if freq:
        oseries = oseries.resample(freq)
    # Drop nan-values form the time series
    oseries.dropna(inplace=True)
    return oseries


def check_tseries(stress, freq, fillnan):
    """ Check the stress series before running a simulation.

    Parameters
    ----------
    tseries: pd.Series
        Pandas series object containing the stress time series.
    freq: str
        String containing the desired frequency. The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: optional: str or float
        Methods or float number to fill nan-values. Default values is
        'mean'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    Returns
    -------
    the corrected stress as pd.Series:
        - Checked for Missing values
        - Checked for frequency of stress
    """

    assert isinstance(stress, pd.Series), 'Expected a Pandas Series, ' \
                                          'got %s' % type(stress)
    # Deal with frequency of the stress series
    if freq:
        stress = stress.asfreq(freq)
    else:
        freq = pd.infer_freq(stress.index)
        stress = stress.asfreq(freq)

    # Deal with nan-values in stress series
    if stress.hasnans:
        print '%i nan-value(s) was/were found and filled with: %s' % (
            stress.isnull().values.sum(), fillnan)
        if fillnan == 'interpolate':
            stress.interpolate('time')
        elif type(fillnan) == float:
            print fillnan, 'init'
            stress.fillna(fillnan, inplace=True)
        else:
            stress.fillna(stress.mean(), inplace=True)  # Default option
    return stress
