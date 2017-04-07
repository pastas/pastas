"""This module is used to check the time series.

"""
from __future__ import print_function, division

import pandas as pd


def check_oseries(oseries, fillnan='drop'):
    """Check the observed time series before running a simulation.

    Parameters
    ----------
    oseries: pd.Series
        Pandas series object containing the observed time series.
    fillnan: optional[str or float]
        Methods or float number to fill nan-values. Default values is
        'drop'. Currently supported options are: 'interpolate', float,
        'mean' and, 'drop'. Interpolation is performed with a standard linear
        interpolation.

    Returns
    -------
    oseries: pd.Series
        Pandas series object checked for nan-values and with the required frequency.

    """
    assert isinstance(oseries, pd.Series), 'Expected a Pandas Series object, ' \
                                           'got %s' % type(oseries)

    # make a deep copy to preserve original imported data
    oseries = oseries.loc[oseries.first_valid_index():oseries.last_valid_index(
    )].copy(deep=True)

    # Handle nan-values in oseries
    if oseries.hasnans:
        print(
            '%i nan-value(s) in the oseries was/were found and handled/filled '
            'with: %s' % (oseries.isnull().values.sum(), fillnan))
        if fillnan == 'drop':
            oseries.dropna(inplace=True)  # Default option
        elif fillnan == 'mean':
            oseries.fillna(oseries.mean(), inplace=True)
        elif fillnan == 'interpolate':
            oseries.interpolate(method='time', inplace=True)
        elif type(fillnan) == float:
            oseries.fillna(fillnan, inplace=True)
        else:
            print(
                'User-defined option for fillnan %s isinstance() not supported'
                % fillnan)

    # Drop dubplicate indexes
    if not oseries.index.is_unique:
        print(
            'duplicate time-indexes were found in the oseries. Values were averaged.')
        grouped = oseries.groupby(level=0)
        oseries = grouped.mean()

    return oseries


def check_tseries(stress, freq, fillnan, name=''):
    """ Check the stress series when creating a time series model.

    Parameters
    ----------
    stress: pd.Series
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

    # Drop nan-values at the beginning and end of the time series
    stress = stress.loc[stress.first_valid_index():stress.last_valid_index(
    )].copy(deep=True)

    # Make sure the indices are Timestamps
    stress.index = pd.to_datetime(stress.index)

    # Make frequency of the stress series constant
    if freq:
        stress = stress.asfreq(freq)
    else:
        freq = pd.infer_freq(stress.index)
        print(
            'Inferred frequency from time series %s: freq=%s ' % (name, freq))
        stress = stress.asfreq(freq)

    # Handle nan-values in stress series
    if stress.hasnans:
        print('%i nan-value(s) was/were found and filled with: %s'
              % (stress.isnull().values.sum(), fillnan))
        if fillnan == 'mean':
            stress.fillna(stress.mean(), inplace=True)  # Default option
        elif fillnan == 'interpolate':
            stress.interpolate(method='time', inplace=True)
        elif fillnan == 'bfill':
            stress.bfill(inplace=True)
        elif type(fillnan) == float:
            stress.fillna(fillnan, inplace=True)
        else:
            print(
                'User-defined option for fillnan %s isinstance() not supported'
                % fillnan)

    return stress
