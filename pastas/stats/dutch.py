"""The following methods are descriptive statistics commonly used to describe
groundwater time series in the Netherlands.

.. codeauthor:: R. Calje, T. van Steijn and R. Collenteur
"""

# Type Hinting
from typing import Optional, Union

from numpy import nan
from packaging.version import parse as parse_version
from pandas import Series, Timedelta, concat, date_range
from pandas import __version__ as pd_version

from pastas.timeseries_utils import get_sample
from pastas.typing import Function, TimestampType

pandas_version = parse_version(pd_version)

year_offset = "YE" if pandas_version >= parse_version("2.2.0") else "A"


def q_ghg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    q: float = 0.94,
    by_year: bool = True,
) -> Series:
    """Gemiddeld Hoogste Grondwaterstand (GHG) also called MHGL (Mean High GW Level).

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GHG for.
    tmin, tmax: pandas.Timestamp, optional
    q : float, optional
        quantile fraction of exceedance (default 0.94)
    by_year: bool, optional
        Take average over quantiles per year (default True)

    Notes
    -----
    Approximated by taking quantiles of the time series values per year and
    calculating the mean of the quantiles. The series is first resampled to daily
    values.
    """
    return _q_gxg(series, q, tmin=tmin, tmax=tmax, by_year=by_year)


def q_glg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    q: float = 0.06,
    by_year: bool = True,
) -> Series:
    """Gemiddeld Laagste Grondwaterstand (GLG) also called MLGL (Mean Low
    Groundwater Level).

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GLG for.
    tmin, tmax: pandas.Timestamp, optional
    q : float, optional
        quantile, fraction of exceedance (default 0.06)
    by_year: bool, optional
        Take average over quantiles per year (default True)

    Notes
    -----
    Approximated by taking quantiles of the time series values per year and
    calculating the mean of the quantiles. The series is first resampled to daily
    values.
    """
    return _q_gxg(series, q, tmin=tmin, tmax=tmax, by_year=by_year)


def q_gvg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    by_year: bool = True,
) -> Series:
    """Gemiddeld Voorjaarsgrondwaterstand (GVG) also called MSGL (Mean Spring GW Level).

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GVG for.
    tmin, tmax: pandas.Timestamp, optional
    by_year: bool, optional
        Take average over quantiles per year (default True)

    Notes
    -----
    Approximated by taking the median of the values in the period between 14 March
    and 15 April (after resampling to daily values). This function does not care
    about series length!
    """
    if tmin is not None:
        series = series.loc[tmin:]
    if tmax is not None:
        series = series.loc[:tmax]
    series = series.resample("d").median()
    inspring = _in_spring(series)
    if any(inspring):
        if by_year:
            return series.loc[inspring].resample(year_offset).median().mean()
        else:
            return series.loc[inspring].median()
    else:
        return nan


def ghg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    fill_method: str = "nearest",
    limit: int = 0,
    output: str = "mean",
    min_n_meas: int = 16,
    min_n_years: int = 8,
    year_offset: str = year_offset + "-MAR",
) -> Union[Series, float]:
    """Calculate the 'Gemiddelde Hoogste Grondwaterstand' (Average High
    Groundwater Level)

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex
        The pandas Series of which the statistic is determined.
    tmin: pandas.Timestamp, optional
        The lowest index to take into account.
    tmax: pandas.Timestamp, optional
        The highest index to take into account.
    fill_method : str
        see .. :mod: pastas.stats.dutch._gxg
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to fill nothing.
    output : str, optional
        output type
        * 'mean' (default) : for mean of yearly values.
        * 'yearly': for series of yearly values.
        * 'g3': for series with selected data for calculating statistic.
        * 'semimonthly': for series with all data points (14th, 28th of each month).
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 24).
    min_n_years: int, optional
        Minimum number of years.
    year_offset: resampling offset. Use 'YE' for calendar years
        (jan 1 to dec 31) and 'YE-MAR' for hydrological years (apr 1 to mar 31).

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values.

    Notes
    -----
    Classic method resampling the series to every 14th and 28th of the month. Taking
    the mean of the mean of three highest values per year.
    """

    # mean_high = lambda s: s.nlargest(3).mean()
    def highs(s, min_n_meas):
        if len(s) < min_n_meas:
            return Series(nan)
        else:
            if len(s) > 20:
                return s.nlargest(3)
            elif len(s) > 12:
                return s.nlargest(2)
            else:
                return s.nlargest(1)

    def mean_high(s, min_n_meas):
        return highs(s, min_n_meas).mean()

    if output in ["mean", "yearly"]:
        f_agg = mean_high
    elif output == "g3":
        f_agg = highs
    elif output == "semimonthly":
        f_agg = None
    else:
        raise ValueError(f"Unrecognized option for output: {output}")

    return _gxg(
        series,
        f_agg,
        tmin=tmin,
        tmax=tmax,
        fill_method=fill_method,
        limit=limit,
        output=output,
        min_n_meas=min_n_meas,
        min_n_years=min_n_years,
        year_offset=year_offset,
    )


def glg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    fill_method: str = "nearest",
    limit: int = 0,
    output: str = "mean",
    min_n_meas: int = 16,
    min_n_years: int = 8,
    year_offset: str = year_offset + "-MAR",
) -> Union[Series, float]:
    """Calculate the 'Gemiddelde Laagste Grondwaterstand' (Average Low GW Level).

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex
        The pandas Series of which the statistic is determined.
    tmin: pandas.Timestamp, optional
        The lowest index to take into account.
    tmax: pandas.Timestamp, optional
        The highest index to take into account.
    fill_method : str, optional
        see .. :mod: pastas.stats.dutch._gxg
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to fill nothing.
    output : str, optional
        output type
        * 'mean' (default) : for mean of yearly values.
        * 'yearly': for series of yearly values.
        * 'g3': for series with selected data for calculating statistic.
        * 'semimonthly': for series with all data points (14th, 28th of each month).
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 24).
    min_n_years: int, optional
        Minimum number of years.
    year_offset: resampling offset. Use 'YE' for calendar years
        (jan 1 to dec 31) and 'YE-MAR' for hydrological years (apr 1 to mar 31).

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values.

    Notes
    -----
    Classic method resampling the series to every 14th and 28th of the month. Taking
    the mean of the mean of three lowest values per year.
    """

    # mean_low = lambda s: s.nsmallest(3).mean()
    def lows(s, min_n_meas):
        if len(s) < min_n_meas:
            return Series(nan)
        else:
            if len(s) > 20:
                return s.nsmallest(3)
            elif len(s) > 12:
                return s.nsmallest(2)
            else:
                return s.nsmallest(1)

    def mean_low(s, min_n_meas):
        return lows(s, min_n_meas).mean()

    if output in ["mean", "yearly"]:
        f_agg = mean_low
    elif output == "g3":
        f_agg = lows
    elif output == "semimonthly":
        f_agg = None
    else:
        raise ValueError(f"Unrecognized option for output: {output}")

    return _gxg(
        series,
        f_agg,
        tmin=tmin,
        tmax=tmax,
        fill_method=fill_method,
        limit=limit,
        output=output,
        min_n_meas=min_n_meas,
        min_n_years=min_n_years,
        year_offset=year_offset,
    )


def gvg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    fill_method: str = "linear",
    limit: int = 8,
    output: str = "mean",
    min_n_meas: int = 2,
    min_n_years: int = 8,
    year_offset: str = year_offset,
) -> Union[Series, float]:
    """Calculate the 'Gemiddelde Voorjaars Grondwaterstand' (Average Spring GW Level).

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex
        The pandas Series of which the statistic is determined.
    tmin: pandas.Timestamp, optional
        The lowest index to take into account.
    tmax: pandas.Timestamp, optional
        The highest index to take into account.
    fill_method : str, optional
        see .. :mod: pastas.stats.dutch._gxg
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to fill nothing.
    output : str, optional
        output type
        * 'mean' (default) : for mean of yearly values.
        * 'yearly': for series of yearly values.
        * 'g3': for series with selected data for calculating statistic.
        * 'semimonthly': for series with all data points (14th, 28th of each month).
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 3).
    min_n_years: int, optional
        Minimum number of years.
    year_offset: resampling offset. Use "YE" for calendar years
        (jan 1 to dec 31) and "YE-MAR" for hydrological years (apr 1 to mar 31).

    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values.

    Notes
    -----
    Classic method resampling the series to every 14th and 28th of the month. Taking
    the mean of the values on March 14, March 28 and April 14.
    """

    def _mean_spring(s, min_n_meas):
        return _get_spring(s, min_n_meas).mean()

    if output in ["mean", "yearly"]:
        f_agg = _mean_spring
    elif output == "g3":
        f_agg = _get_spring
    elif output == "semimonthly":
        f_agg = None
    else:
        raise ValueError(f"Unrecognized option for output: {output}")

    return _gxg(
        series,
        f_agg,
        tmin=tmin,
        tmax=tmax,
        fill_method=fill_method,
        limit=limit,
        output=output,
        min_n_meas=min_n_meas,
        min_n_years=min_n_years,
        year_offset=year_offset,
    )


def gg(
    series: Series,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    fill_method: str = "nearest",
    limit: int = 0,
    output: str = "mean",
    min_n_meas: int = 16,
    min_n_years: int = 8,
    year_offset: str = year_offset + "-MAR",
) -> Union[Series, float]:
    """Calculate the 'Gemiddelde Grondwaterstand' (Average Groundwater Level).

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex
        The pandas Series of which the statistic is determined.
    tmin: pandas.Timestamp, optional
        The lowest index to take into account.
    tmax: pandas.Timestamp, optional
        The highest index to take into account.
    fill_method : str, optional
        see .. :mod: pastas.stats.dutch._gxg
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to fill nothing.
    output : str, optional
        output type
        * 'mean' (default) : for mean of yearly values.
        * 'yearly': for series of yearly values.
        * 'g3': for series with selected data for calculating statistic.
        * 'semimonthly': for series with all data points (14th, 28th of each month).
    min_n_meas: int, optional
        Minimum number of measurements per year (at maximum 24).
    min_n_years: int, optional
        Minimum number of years.
    year_offset: resampling offset. Use "YE" for calendar years (jan 1 to dec 31) and
    'YE-MAR' for hydrological years (apr 1 to mar 31).

    Returns
    -------
    pd.Series or scalar
        series of yearly values or mean of yearly values.

    Notes
    -----
    Classic method resampling the series to every 14th and 28th of the month.
    """

    # mean_low = lambda s: s.nsmallest(3).mean()
    def mean_all(s, min_n_meas):
        if len(s) < min_n_meas:
            return nan
        else:
            return s.mean()

    return _gxg(
        series,
        mean_all,
        tmin=tmin,
        tmax=tmax,
        fill_method=fill_method,
        limit=limit,
        output=output,
        min_n_meas=min_n_meas,
        min_n_years=min_n_years,
        year_offset=year_offset,
    )


# Helper functions


def _get_spring(series: Series, min_n_meas: int) -> float:
    """Internal method to get values of time series values in spring.

    Part of year aggregator function for gvg method.

    Parameters
    ----------
    series : pandas.Series
        series with datetime index.

    Returns
    -------
    float
        values of series in spring, or NaN if no values in spring.
    """
    inspring = _in_spring(series)
    if inspring.sum() < min_n_meas:
        return Series(nan)
    else:
        return series.loc[inspring]


def _in_spring(series: Series) -> Series:
    """Internal method to test if time series index is between 14 March and 15 April.

    Parameters
    ----------
    series : pd.Series
        series with datetime index.

    Returns
    -------
    pd.Series
        Boolean series with datetimeindex.
    """

    def isinspring(x):
        return ((x.month == 3) and (x.day >= 14)) or ((x.month == 4) and (x.day < 15))

    return Series([isinspring(x) for x in series.index], index=series.index)


def _gxg(
    series: Series,
    year_agg: Function,
    tmin: Optional[TimestampType],
    tmax: Optional[TimestampType],
    fill_method: str,
    limit: Union[int, None],
    output: str,
    min_n_meas: int,
    min_n_years: int,
    year_offset: str,
) -> Union[Series, float]:
    """Internal method for classic GXG statistics. Resampling the series to every
    14th and 28th of the month. Taking the mean of aggregated values per year.

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex
        The pandas Series of which the statistic is determined.
    year_agg : function series -> scalar
        Aggregator function to one value per year.
    tmin: pandas.Timestamp, optional
        The lowest index to take into account.
    tmax: pandas.Timestamp, optional
        The highest index to take into account.
    fill_method : str
        see notes below.
    limit : int or None, optional
        Maximum number of days to fill using fill method, use None to fill nothing.
    output : str
        output type
        * 'mean' (default) : for mean of yearly values
        * 'yearly': for series of yearly values
        * 'g3': for series with selected data for calculating statistic
        * 'semimonthly': for series with all data points (14th, 28th of each month)
    min_n_meas: int, optional
        Minimum number of measurements per year.
    min_n_years: int
        Minimum number of years.
    year_offset: string
        resampling offset. Use "YE" for calendar years (jan 1 to dec 31) and
        'YE-MAR' for hydrological years (apr 1 to mar 31)


    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values.

    Raises
    ------
    ValueError
        When output argument is unknown.

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
        if output.startswith("year"):
            return Series()
        elif output == "mean":
            return nan
        else:
            ValueError("{output:} is not a valid output option".format(output=output))

    # resample the series to values at the 14th and 28th of every month
    # first generate a daily series by averaging multiple measurements during the day
    series = series.resample("d").mean()
    select14or28 = True
    if fill_method is None:
        series = series.dropna()
    elif fill_method == "ffill":
        series = series.ffill(limit=limit)
    elif fill_method == "bfill":
        series = series.bfill(limit=limit)
    elif fill_method == "nearest":
        if limit == 0:
            # limit=0 is a trick to only use each measurement once
            # only keep days with measurements
            series = series.dropna()
            # generate an index at the 14th and 28th of every month
            buf = Timedelta(8, "d")
            ref_index = date_range(series.index.min() - buf, series.index.max() + buf)
            mask = [(x.day == 14) or (x.day == 28) for x in ref_index]
            ref_index = ref_index[mask]
            # only keep the days that are closest to series.index
            ref_index = get_sample(ref_index, series.index)
            # and set the index of series to this index
            # (and remove rows in series that are not in ref_index)
            series = series.reindex(ref_index, method=fill_method)
            select14or28 = False
        else:
            # with a large limit (larger than 6) it is possible that one measurement
            # is used more than once
            series = series.dropna().reindex(
                series.index, method=fill_method, limit=limit
            )
    else:
        series = series.interpolate(
            method=fill_method, limit=limit, limit_direction="both"
        )

    # and select the 14th and 28th of each month (if needed still)
    if select14or28:
        mask = [(x.day == 14) or (x.day == 28) for x in series.index]
        series = series.loc[mask]

    # remove NaNs that may have formed in the process above
    series.dropna(inplace=True)

    # resample the series to yearly values
    if output == "semimonthly":
        return series
    elif output in ["yearly", "mean"]:
        yearly = series.resample(year_offset).apply(year_agg, min_n_meas=min_n_meas)
    elif output == "g3":
        yearly = series.resample(year_offset)
        collect = {}
        for yr, group in yearly:
            s = year_agg(group, min_n_meas=min_n_meas)
            if isinstance(s, Series):
                s = s.sort_index()
            collect[yr] = s
        yearly = concat(collect)

    # return statements
    if output.startswith("year"):
        return yearly
    elif output == "g3":
        return yearly
    elif output == "mean":
        if yearly.notna().sum() < min_n_years:
            return nan
        else:
            return yearly.mean()
    else:
        msg = "{} is not a valid output option".format(output)
        raise (ValueError(msg))


def _q_gxg(
    series: Series,
    q: float,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    by_year: bool = True,
) -> Series:
    """Dutch groundwater statistics GHG and GLG approximated by taking quantiles of
    the time series values per year and taking the mean of the quantiles.

    The series is first resampled to daily values.

    Parameters
    ----------
    series: pandas.Series
        Series to calculate the GXG for.
    q: float
        quantile fraction of exceedance.
    tmin: pandas.Timestamp, optional
    tmax: pandas.Timestamp, optional
    by_year: bool, optional
        Take average over quantiles per year (default True).
    """
    if tmin is not None:
        series = series.loc[tmin:]
    if tmax is not None:
        series = series.loc[:tmax]
    series = series.resample("d").median()
    if by_year:
        return series.resample(year_offset).apply(lambda s: s.quantile(q)).mean()
    else:
        return series.quantile(q)
