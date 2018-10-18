"""This module contains a set of Dutch Statistics which are commonly used in
groundwater statistics.

Authors: R. Calje, T. van Steijn and R. Collenteur

"""

from numpy import nan, any
from pandas import date_range, Series, to_timedelta

from ..utils import get_sample


# %% Some Dutch statistics
def q_ghg(series, tmin=None, tmax=None, q=0.94, by_year=True):
    """Gemiddeld Hoogste Grondwaterstand (GHG) also called MHGL (Mean High
    Groundwater Level). Approximated by taking quantiles of the
    timeseries values per year and calculating the mean of the quantiles.

    The series is first resampled to daily values.

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GHG for.
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    q : float, optional
        quantile fraction of exceedance (default 0.94)
    by_year: bool, optional
        Take average over quantiles per year (default True)
    """
    return __q_gxg__(series, q, tmin=tmin, tmax=tmax, by_year=by_year)


def q_glg(series, tmin=None, tmax=None, q=0.06, by_year=True):
    """Gemiddeld Laagste Grondwaterstand (GLG) also called MLGL (Mean Low
    Groundwater Level). Approximated by taking quantiles of the
    timeseries values per year and calculating the mean of the quantiles.

    The series is first resampled to daily values.

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GLG for.
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    q : float, optional
        quantile, fraction of exceedance (default 0.06)
    by_year: bool, optional
        Take average over quantiles per year (default True)
    """
    return __q_gxg__(series, q, tmin=tmin, tmax=tmax, by_year=by_year)


def q_gvg(series, tmin=None, tmax=None, by_year=True):
    """Gemiddeld Voorjaarsgrondwaterstand (GVG) also called MSGL (Mean
    Spring Groundwater Level) approximated by taking the median of the
    values in the period between 14 March and 15 April (after resampling to
    daily values).

    This function does not care about series length!

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GVG for.
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    by_year: bool, optional
        Take average over quantiles per year (default True)
    """
    if tmin is not None:
        series = series.loc[tmin:]
    if tmax is not None:
        series = series.loc[:tmax]
    series = series.resample('d').median()
    inspring = __in_spring__(series)
    if any(inspring):
        if by_year:
            return (series
                    .loc[inspring]
                    .resample('a')
                    .median()
                    .mean()
                    )
        else:
            return series.loc[inspring].median()
    else:
        return nan


def ghg(series, tmin=None, tmax=None, fill_method='nearest', limit=0,
        output='mean', min_n_meas=16, min_n_years=8, year_offset='a-mar'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the mean of three highest values per
    year.

    Parameters
    ----------
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    series
    fill_method : str
        see .. :mod: pastas.stats.__gxg__
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to
        fill nothing
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for mean
        of yearly values
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 24).
    min_n_years: int, optional
        Minimum number of years
    year_offset: resampling offset. Use 'a' for calendar years
        (jan 1 to dec 31) and 'a-mar' for hydrological years (apr 1 to mar 31)

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values

    """

    # mean_high = lambda s: s.nlargest(3).mean()
    def mean_high(s, min_n_meas):
        if len(s) < min_n_meas:
            return nan
        else:
            if len(s) > 20:
                return s.nlargest(3).mean()
            elif len(s) > 12:
                return s.nlargest(2).mean()
            else:
                return s.nlargest(1).mean()

    return __gxg__(series, mean_high, tmin=tmin, tmax=tmax,
                   fill_method=fill_method, limit=limit, output=output,
                   min_n_meas=min_n_meas, min_n_years=min_n_years,
                   year_offset=year_offset)


def glg(series, tmin=None, tmax=None, fill_method='nearest', limit=0,
        output='mean', min_n_meas=16, min_n_years=8, year_offset='a-mar'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the mean of three lowest values per year.

    Parameters
    ----------
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    series
    fill_method : str, optional
        see .. :mod: pastas.stats.__gxg__
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to
        fill nothing.
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 24)
    min_n_years: int, optional
        Minimum number of years
    year_offset: resampling offset. Use 'a' for calendar years
        (jan 1 to dec 31) and 'a-mar' for hydrological years (apr 1 to mar 31)

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values

    """

    # mean_low = lambda s: s.nsmallest(3).mean()
    def mean_low(s, min_n_meas):
        if len(s) < min_n_meas:
            return nan
        else:
            if len(s) > 20:
                return s.nsmallest(3).mean()
            elif len(s) > 12:
                return s.nsmallest(2).mean()
            else:
                return s.nsmallest(1).mean()

    return __gxg__(series, mean_low, tmin=tmin, tmax=tmax,
                   fill_method=fill_method, limit=limit, output=output,
                   min_n_meas=min_n_meas, min_n_years=min_n_years,
                   year_offset=year_offset)


def gvg(series, tmin=None, tmax=None, fill_method='linear', limit=8,
        output='mean', min_n_meas=2, min_n_years=8, year_offset='a'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the values on March 14, March 28 and
    April 14.

    Parameters
    ----------
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    series
    fill_method : str, optional
        see .. :mod: pastas.stats.__gxg__
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to
        fill nothing
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 3)
    min_n_years: int, optional
        Minimum number of years
    year_offset: resampling offset. Use 'a' for calendar years
        (jan 1 to dec 31) and 'a-mar' for hydrological years (apr 1 to mar 31)

    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values

    """
    return __gxg__(series, __mean_spring__, tmin=tmin, tmax=tmax,
                   fill_method=fill_method, limit=limit, output=output,
                   min_n_meas=min_n_meas, min_n_years=min_n_years,
                   year_offset=year_offset)


# Helper functions

def __mean_spring__(series, min_n_meas):
    """Internal method to determine mean of timeseries values in spring.

    Year aggregator function for gvg method.

    Parameters
    ----------
    series : pandas.Series
        series with datetime index

    Returns
    -------
    float
        Mean of series, or NaN if no values in spring

    """
    inspring = __in_spring__(series)
    if inspring.sum() < min_n_meas:
        return nan
    else:
        return series.loc[inspring].mean()


def __in_spring__(series):
    """Internal method to test if timeseries index is between 14 March and 15
    April.

    Parameters
    ----------
    series : pd.Series
        series with datetime index

    Returns
    -------
    pd.Series
        Boolean series with datetimeindex
    """
    isinspring = lambda x: (((x.month == 3) and (x.day >= 14)) or
                            ((x.month == 4) and (x.day < 15)))
    return Series(series.index.map(isinspring), index=series.index)


def __gxg__(series, year_agg, tmin, tmax, fill_method, limit, output,
            min_n_meas, min_n_years, year_offset):
    """Internal method for classic GXG statistics. Resampling the series to
    every 14th and 28th of the month. Taking the mean of aggregated
    values per year.

    Parameters
    ----------
    year_agg : function series -> scalar
        Aggregator function to one value per year
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    fill_method : str
        see notes below
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to
        fill nothing
    output : str
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values
    min_n_meas: int, optional
        Minimum number of measurements per year
    min_n_years: int
        Minimum number of years.
    year_offset: string
        resampling offset. Use 'a' for calendar years (jan 1 to dec 31)
        and 'a-mar' for hydrological years (apr 1 to mar 31)


    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values

    Raises
    ------
    ValueError
        When output argument is unknown

    Notes
    -----
    fill method for interpolation to 14th and 28th of the month see:
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.ffill.html
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.bfill.html
        * https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.reindex.html
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
        * Use None to omit filling and drop NaNs

    """
    # handle tmin and tmax
    if tmin is not None:
        series = series.loc[tmin:]
    if tmax is not None:
        series = series.loc[:tmax]
    if series.empty:
        if output.startswith('year'):
            return Series()
        elif output == 'mean':
            return nan
        else:
            ValueError('{output:} is not a valid output option'.format(
                output=output))

    # resample the series to values at the 14th and 28th of every month
    # first generate a daily series by averaging multiple measurements during the day
    series = series.resample('d').mean()
    select14or28 = True
    if fill_method is None:
        series = series.dropna()
    elif fill_method == 'ffill':
        series = series.ffill(limit=limit)
    elif fill_method == 'bfill':
        series = series.bfill(limit=limit)
    elif fill_method == 'nearest':
        if limit == 0:
            # limit=0 is a trick to only use each measurements once
            # only keep days with measurements
            series = series.dropna()
            # generate an index at the 14th and 28th of every month
            buf = to_timedelta(8, 'd')
            ref_index = date_range(series.index.min() - buf,
                                   series.index.max() + buf)
            mask = [(x.day == 14) or (x.day == 28) for x in ref_index]
            ref_index = ref_index[mask]
            # only keep the days that are closest to series.index
            ref_index = get_sample(ref_index, series.index)
            # and set the index of series to this index
            # (and remove rows in series that are not in ref_index)
            series = series.reindex(ref_index, method=fill_method)
            select14or28 = False
        else:
            # with a large limit (larger than 6) it is possible that one measurement is used more than once
            series = series.dropna().reindex(series.index, method=fill_method,
                                             limit=limit)
    else:
        series = series.interpolate(method=fill_method, limit=limit,
                                    limit_direction='both')

    # and select the 14th and 28th of each month (if needed still)
    if select14or28:
        mask = [(x.day == 14) or (x.day == 28) for x in series.index]
        series = series.loc[mask]

    # remove NaNs that may have formed in the process above
    series.dropna(inplace=True)

    # resample the series to yearly values
    yearly = series.resample(year_offset).apply(year_agg,
                                                min_n_meas=min_n_meas)

    # return statements
    if output.startswith('year'):
        return yearly
    elif output == 'mean':
        if yearly.notna().sum() < min_n_years:
            return nan
        else:
            return yearly.mean()
    else:
        ValueError('{output:} is not a valid output option'.format(
            output=output))


def __q_gxg__(series, q, tmin=None, tmax=None, by_year=True):
    """Dutch groundwater statistics GHG and GLG approximated
    by taking quantiles of the timeseries values per year
    and taking the mean of the quantiles.

    The series is first resampled to daily values.

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GXG for.
    q: float
        quantile fraction of exceedance
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    by_year: bool, optional
        Take average over quantiles per year (default True)
    """
    if tmin is not None:
        series = series.loc[tmin:]
    if tmax is not None:
        series = series.loc[:tmax]
    series = series.resample('d').median()
    if by_year:
        return (series
                .resample('a')
                .apply(lambda s: s.quantile(q))
                .mean()
                )
    else:
        return series.quantile(q)
