import pandas as pd


def check_oseries(oseries, freq, fillnan='drop'):
    """Check the observed time series before running a simulation.

    Parameters
    ----------
    oseries: pd.Series
        Pandas series object containing the observed time series.
    freq: str
        String containing the desired frequency. The required string format is found
        at http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset
        -aliases
    fillnan: optional: str or float
        Methods or float number to fill nan-values. Default values is
        'drop'. Currently supported options are: 'interpolate', float,
        and 'mean'. Interpolation is performed with a standard linear
        interpolation.

    Returns
    -------
    oseries: pd.Series
        Pandas series object check for nan-values and with the required frequency.

    """
    assert isinstance(oseries, pd.Series), 'Expected a Pandas Series object, ' \
                                           'got %s' % type(oseries)

    # make a deep copy to preserve original imported data
    oseries = oseries.loc[oseries.first_valid_index():oseries.last_valid_index(
    )].copy(deep=True)

    # Deal with frequency of the time series
    if freq:
        oseries = oseries.resample(freq)

    # Handle nan-values in oseries
    if oseries.hasnans:
        print '%i nan-value(s) was/were found and handled/filled with: %s' % (
            oseries.isnull().values.sum(), fillnan)
        if fillnan == 'mean':
            oseries.fillna(stress.mean(), inplace=True)
        elif fillnan == 'interpolate':
            oseries.interpolate(method='time', inplace=True)
        elif type(fillnan) == float:
            oseries.fillna(fillnan, inplace=True)
        else:
            oseries.dropna(inplace=True) # Default option

    return oseries


def check_tseries(stress, freq, fillnan):
    """ Check the stress series when creating a time series model.

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
        - nan-values dropped at begin and end.
        - frequency made constant.
        - handled nan-values in between.
    """

    assert isinstance(stress, pd.Series), 'Expected a Pandas Series, ' \
                                          'got %s' % type(stress)

    # Drop nan-values at the beginning of the time series
    stress = stress.loc[stress.first_valid_index():stress.last_valid_index(
    )].copy(deep=True)

    # Make frequency of the stress series constant
    if freq:
        stress = stress.asfreq(freq)
    else:
        freq = pd.infer_freq(stress.index)
        print 'Tried to infer frequency from time series: freq=%s' %freq
        stress = stress.asfreq(freq)

    # Handle nan-values in stress series
    if stress.hasnans:
        print '%i nan-value(s) was/were found and filled with: %s' % (
            stress.isnull().values.sum(), fillnan)
        if fillnan == 'interpolate':
            stress.interpolate(method='time')
        elif type(fillnan) == float:
            print fillnan, 'init'
            stress.fillna(fillnan, inplace=True)
        else:
            stress.fillna(stress.mean(), inplace=True)  # Default option
    return stress
