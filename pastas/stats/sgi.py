"""This module contains methods to compute the Standardized Groundwater Index."""

from numpy import array, linspace
from pandas import DataFrame, Series
from scipy.stats import norm


def sgi(series: Series, timescale_months: int = 1) -> Series:
    """Method to compute the Standardized Groundwater Index (SGI)
    :cite:t:`bloomfield_analysis_2013`.

    Parameters
    ----------
    series: pandas.Series or Pandas.DataFrame
        Pandas time series of the groundwater levels
        for which the SGI is to be determined
    timescale_months: integer, optional
        Length of the aggregation period in months (default: 1; allowed: 1, 2, 3)

    Returns
    -------
    sgi_series: pandas.Series or Pandas.DataFrame
        Pandas time series of the groundwater levels. Time series index should be a
        pandas DatetimeIndex.

    Notes
    -----
    The Standardized Groundwater Index (SGI) is a non-parametric method to
    standardize groundwater levels. The SGI is calculated for each aggregation
    period within the year separately. The data within that period in all years
    in the series are used to determine the reference for which the index is
    calculated for each value in that period.
    The SGI is a dimensionless index and is used to compare groundwater levels
    across different wells. It may be useful to resample the time series to a
    monthly interval before computing the SGI.
    """
    if timescale_months not in (1, 2, 3):
        raise ValueError(
            "SGI can only be called with timescale_months = 1, 2, or 3; not"
            + str(timescale_months)
        )
    if isinstance(series, DataFrame):
        series = series.apply(sgi, timescale_months=timescale_months)
    elif isinstance(series, Series):
        # Create a copy to ensure series is untouched.
        # Set dtype to avoid conflict when assigning SGI values
        series = series.copy().dropna().astype(float)

        # Loop over the months
        for month in range(1, 13, timescale_months):
            sel = array(range(timescale_months)) + month
            data = series[series.index.month.isin(sel)]
            n = data.size  # Number of observations
            pmin = 1 / (2 * n)
            pmax = 1 - pmin
            sgi_values = norm.ppf(linspace(pmin, pmax, n))
            series.loc[data.sort_values().index] = sgi_values

    return series
