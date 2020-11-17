"""This module contains methods to compute the Standardized Groundwater Index.

"""

from numpy import linspace
from scipy.stats import norm


def sgi(series):
    """Method to compute the Standardized Groundwater Index.

    Parameters
    ----------
    series: pandas.Series

    Returns
    -------
    sgi_series: pandas.Series
        Pandas time series of the groundwater levels. Time series index
        should be a pandas DatetimeIndex.

    References
    ----------
    .. [sgi_2013]: Bloomfield, J. P. and Marchant, B. P.: Analysis of
       groundwater drought building on the standardised precipitation index
       approach, Hydrol. Earth Syst. Sci., 17, 4769–4787, 2013.

    """
    series = series.copy()  # Create a copy to ensure series is untouched.

    # Loop over the months
    for month in range(1, 13):
        data = series[series.index.month == month]
        n = data.size  # Number of observations
        pmin = 1 / (2 * n)
        pmax = 1 - pmin
        sgi_values = norm.ppf(linspace(pmin, pmax, n))
        series.loc[data.sort_values().index] = sgi_values
    return series
