import numpy as np
import pandas as pd


def check_oseries(oseries, freq):
    assert isinstance(oseries, pd.Series), 'Expected a Pandas Series object, ' \
                                           'got %s' % type(oseries)
    # Deal with frequency of the time series
    if freq:
        oseries = oseries.resample(freq)
    # Drop nan-values form the time series
    oseries.dropna(inplace=True)
    return oseries


def check_tseries(stress, freq, fillna):
    """ Check the stress series on missing values and constant frequency.

    Returns
    -------
    list of stresses:
        - Checked for Missing values
        - Checked for frequency of stress
    """

    if type(stress) is pd.Series:
        stress = [stress]
    stresses = []
    for k in stress:
        assert isinstance(k, pd.Series), 'Expected a Pandas Series, ' \
                                         'got %s' % type(k)
        # Deal with frequency of the stress series
        if freq:
            k = k.asfreq(freq)
        else:
            freq = pd.infer_freq(k.index)
            k = k.asfreq(freq)

        # Deal with nan-values in stress series
        if k.hasnans:
            print '%i nan-value(s) was/were found and filled with: %s' % (
                k.isnull(
                ).values.sum(), fillna)
            if fillna == 'interpolate':
                k.interpolate('time')
            elif type(fillna) == float:
                print fillna, 'init'
                k.fillna(fillna, inplace=True)
            else:
                k.fillna(k.mean(), inplace=True)  # Default option
        stresses.append(k)
    return stresses
