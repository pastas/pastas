"""This module contains methods to compute the Standardized Groundwater Index."""

from numpy import linspace
from pandas import DataFrame, Series
from scipy.stats import norm


def sgi(series: Series) -> Series:
    """Method to compute the Standardized Groundwater Index
    :cite:t:`bloomfield_analysis_2013`.

    Parameters
    ----------
    series: pandas.Series or Pandas.DataFrame

    Returns
    -------
    sgi_series: pandas.Series or Pandas.DataFrame
        Pandas time series of the groundwater levels. Time series index should be a
        pandas DatetimeIndex.

    Notes
    -----
    The Standardized Groundwater Index (SGI) is a non-parametric method to
    standardize groundwater levels. The SGI is calculated for each month
    separately. The SGI is a dimensionless index and is used to compare
    groundwater levels across different wells. It is generally recommended to resample
    the data to a monthly time series before computing the SGI.

    """
    if isinstance(series, DataFrame):
        series = series.apply(sgi)
    elif isinstance(series, Series):
        series = series.dropna().copy()  # Create a copy to ensure series is untouched.

        # Loop over the months
        for month in range(1, 13):
            data = series[series.index.month == month]
            n = data.size  # Number of observations
            pmin = 1 / (2 * n)
            pmax = 1 - pmin
            sgi_values = norm.ppf(linspace(pmin, pmax, n))
            series.loc[data.sort_values().index] = sgi_values

    return series
