"""This module contains methods to compute the groundwater signatures. Part of the
signatures selection is based on the work of :cite:t:`heudorfer_index-based_2019`."""

# Type Hinting
from logging import getLogger
from typing import Optional, Tuple, Union

from numpy import (
    arctan,
    array,
    cos,
    diff,
    exp,
    isclose,
    isnan,
    linspace,
    log,
    nan,
    ndarray,
    pi,
    sin,
    split,
    sqrt,
    where,
)
from packaging.version import parse as parse_version
from pandas import DataFrame, DatetimeIndex, Series, Timedelta, concat, cut, to_datetime
from pandas import __version__ as pd_version
from scipy.optimize import curve_fit
from scipy.stats import linregress

import pastas as ps
from pastas.stats.core import acf

pandas_version = parse_version(pd_version)

year_offset = "YE" if pandas_version >= parse_version("2.2.0") else "A"

month_offset = "ME" if pandas_version >= parse_version("2.2.0") else "M"

__all__ = [
    "cv_period_mean",
    "cv_date_min",
    "cv_date_max",
    "cv_fall_rate",
    "cv_rise_rate",
    "parde_seasonality",
    "avg_seasonal_fluctuation",
    "interannual_variation",
    "low_pulse_count",
    "high_pulse_count",
    "low_pulse_duration",
    "high_pulse_duration",
    "bimodality_coefficient",
    "mean_annual_maximum",
    "rise_rate",
    "fall_rate",
    "reversals_avg",
    "reversals_cv",
    "colwell_contingency",
    "colwell_constancy",
    "recession_constant",
    "recovery_constant",
    "duration_curve_slope",
    "duration_curve_ratio",
    "richards_pathlength",
    "baselevel_index",
    "baselevel_stability",
    "magnitude",
    "autocorr_time",
    "date_min",
    "date_max",
]

logger = getLogger(__name__)


def _normalize(series: Series) -> Series:
    """Normalize the time series by subtracting the mean and dividing over the range.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series to be normalized.

    Returns
    -------
    series: pandas.Series
        Pandas Series scaled by subtracting the mean and dividing over the range of the
        values. This results in a time series with values between zero and one.

    """
    series = (series - series.min()) / (series.max() - series.min())
    return series


def cv_period_mean(
    series: Series, normalize: bool = False, freq: str = month_offset
) -> float:
    """Coefficient of variation of the mean head over a period (default monthly).

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    freq: str, optional
        frequency to resample the series to by averaging.

    Returns
    -------
    cv: float
        Coefficient of variation of mean head resampled over a period (default monthly).

    Notes
    -----
    Coefficient of variation of mean monthly heads, adapted after
    :cite:t:`hughes_hydrological_1989`. The higher the coefficient of variation, the
    more variable the mean monthly head is throughout the year, and vice versa. The
    coefficient of variation is the standard deviation divided by the mean.

    Examples
    --------
    >>> import pandas as pd
    >>> from pastas.stats.signatures import cv_period_mean
    >>> series = pd.Series([1, 2, 3, 4, 5, 6],
                        index=pd.date_range(start='2022-01-01', periods=6, freq='M'))
    >>> cv = cv_period_mean(series)
    >>> print(cv)

    """
    if normalize:
        series = _normalize(series)

    series = series.resample(freq).mean()
    cv = series.std(ddof=1) / series.mean()  # ddof=1 = > sample std
    return cv


def _cv_date_min_max(series: Series, stat: str) -> float:
    """Method to compute the coefficient of variation of the date of annual
    minimum or maximum head using circular statistics.

    Parameters
    ----------
    series : Series
        Pandas Series with DatetimeIndex and head values.
    stat : str
        "min" or "max" to compute the cv of the date of the annual minimum or maximum
        head.

    Returns
    -------
    float:
        Circular coefficient of variation of the date of annual minimum or maximum
        head.

    Notes
    -----
    Coefficient of variation of the date of annual minimum or maximum head computed
    using circular statistics as described in :cite:t:`fisher_statistical_1995` (page
    32). If there are multiple dates with the same minimum or maximum head, the first
    date is chosen. The higher the coefficient of variation, the more variable the date
    of the annual minimum or maximum head is, and vice versa.

    """
    if stat == "min":
        data = series.groupby(series.index.year).idxmin(skipna=True).dropna().values
    elif stat == "max":
        data = series.groupby(series.index.year).idxmax(skipna=True).dropna().values

    doy = DatetimeIndex(data).dayofyear.to_numpy(float)

    m = 365.25
    two_pi = 2 * pi

    thetas = array(doy) * two_pi / m
    c = cos(thetas).sum()
    s = sin(thetas).sum()
    r = sqrt(c**2 + s**2) / doy.size

    if (s > 0) & (c > 0):
        mean_theta = arctan(s / c)
    elif c < 0:
        mean_theta = arctan(s / c) + pi
    elif (s < 0) & (c > 0):
        mean_theta = arctan(s / c) + two_pi
    else:
        # This should never happen
        raise ValueError("Something went wrong in the circular statistics.")

    mu = mean_theta * m / two_pi
    std = sqrt(-2 * log(r)) * m / two_pi
    return std / mu


def cv_date_min(series: Series) -> float:
    """Coefficient of variation of the date of annual minimum head.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    cv: float
        Coefficient of variation of the date of annual minimum head.

    Notes
    -----
    Coefficient of variation of the date of annual minimum head computed using circular
    statistics as described in :cite:t:`fisher_statistical_1995` (page 32). If there
    are multiple dates with the same minimum head, the first date is chosen. The higher
    the coefficient of variation, the more variable the date of the annual minimum head
    is, and vice versa.

    """
    cv = _cv_date_min_max(series, stat="min")
    return cv


def cv_date_max(series: Series) -> float:
    """Coefficient of variation of the date of annual maximum head.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    cv: float
        Coefficient of variation of the date of annual maximum head.

    Notes
    -----
    Coefficient of variation of the date of annual maximum head computed using circular
    statistics as described in :cite:t:`fisher_statistical_1995` (page 32). If there
    are multiple dates with the same maximum head, the first date is chosen. The higher
    the coefficient of variation, the more variable the date of the maximum head is,
    and vice versa.

    """
    cv = _cv_date_min_max(series, stat="max")
    return cv


def parde_seasonality(series: Series, normalize: bool = True) -> float:
    """Parde seasonality according to :cite:t:`parde_fleuves_1933`, adapted for heads.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Parde seasonality.

    Notes
    -----
    Pardé seasonality is the difference between the maximum and minimum Pardé
    coefficient. A Pardé series consists of 12 Pardé coefficients, corresponding to
    12 months. Pardé coefficient for, for example, January is its long-term monthly
    mean head divided by the overall mean head. The higher the Pardé seasonality, the
    more seasonal the head is, and vice versa.

    """
    coefficients = _parde_coefficients(series=series, normalize=normalize)
    return coefficients.max() - coefficients.min()


def _parde_coefficients(series: Series, normalize: bool = True) -> Series:
    """Parde coefficients for each month :cite:t:`parde_fleuves_1933`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    coefficients: pandas.Series
        Parde coefficients for each month.

    Notes
    -----
    Pardé seasonality is the difference between the maximum and minimum Pardé
    coefficient. A Pardé series consists of 12 Pardé coefficients, corresponding to
    12 months. Pardé coefficient for, for example, January is its long-term monthly
    mean head divided by the overall mean head.

    Examples
    --------
    >>> import pandas as pd
    >>> from pastas.stats.signatures import parde_coefficients
    >>> series = pd.Series([1, 2, 3, 4, 5, 6],
                        index=pd.date_range(start='2022-01-01', periods=6, freq='M'))
    >>> coefficients = parde_coefficients(series)
    >>> print(coefficients)
    month
    1    0.0
    2    0.4
    3    0.8
    4    1.2
    5    1.6
    6    2.0
    dtype: float64

    """
    if normalize:
        series = _normalize(series)

    coefficients = series.groupby(series.index.month).mean() / series.mean()
    coefficients.index.name = "month"
    return coefficients


def _martens(series: Series, normalize: bool = False) -> Tuple[Series, Series]:
    """Function for the average seasonal fluctuation and interannual fluctuation.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    hl: pandas.Series
        Average of the three lowest heads in a year.
    hw: pandas.Series
        Average of the three largest heads in a year.

    Notes
    -----
    According to :cite:t:`martens_groundwater_2013`. The average of the three lowest
    and three highest heads in three different months for each year is computed. The
    average is then taken over all years.

    """
    if normalize:
        series = _normalize(series)

    s = series.resample(month_offset)
    s_min = s.min()
    s_max = s.max()
    hl = s_min.groupby(s_min.index.year).nsmallest(3).groupby(level=0).mean()
    hw = s_max.groupby(s_max.index.year).nlargest(3).groupby(level=0).mean()

    return hl, hw


def avg_seasonal_fluctuation(series: Series, normalize: bool = False) -> float:
    """Average seasonal fluctuation after :cite:t:`martens_groundwater_2013`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Average seasonal fluctuation (s).

    Notes
    -----
    Mean annual difference between the averaged 3 highest monthly heads
    per year and the averaged 3 lowest monthly heads per year.

    Average seasonal fluctuation (s):

        s = MHW - MLW

    A higher value of s indicates a more seasonal head, and vice versa.

    Warning
    -------
    In this formulating the water table is referenced to a certain datum and
    positive, not as depth below the surface!

    """

    hl, hw = _martens(series, normalize=normalize)
    return (hw - hl).mean()


def interannual_variation(series: Series, normalize: bool = False) -> float:
    """Interannual variation after :cite:t:`martens_groundwater_2013`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Interannual variation (s).

    Notes
    -----
    The average between the range in annually averaged 3 highest monthly heads and the
    range in annually averaged 3 lowest monthly heads.

    Inter-yearly variation of high and low water table (s):

        s = ((max_HW - min_HW) + (max_LW - min_LW)) / 2

    A higher value of s indicates a more variable head, and vice versa.

    Warning
    -------
    In this formulating the water table is referenced to a certain datum and
    positive, not as depth below the surface!

    """

    hl, hw = _martens(series, normalize=normalize)
    return ((hw.max() - hw.min()) + (hl.max() - hl.min())) / 2


def _colwell_components(
    series: Series,
    bins: int = 11,
    freq: str = "W",
    method: str = "mean",
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """Colwell's predictability, constant, and contingency
    :cite:t:`colwell_predictability_1974`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int
        number of bins to determine the states of the groundwater.
    freq: str, optional
        frequency to resample the series to. Possible options are "D", "W", "M" or "ME".
    method: str, optional
        Method to use for resampling. Only "mean" is allowed now.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    p, c, m: float, float, float
        predictability, constancy, contingency

    Notes
    -----
    The difference between the sum of entropy for each time step and possible state
    of the seasonal cycle, and the overall entropy across all states and time steps,
    divided by the logarithm of the absolute number of possible states. Entropy
    according to definition in information theory, see reference for details.

    """
    # Prepare data and pivot table
    if normalize:
        series = _normalize(series)

    if method == "mean":
        series = series.resample(freq).mean().dropna()
    else:
        raise NotImplementedError

    series.name = "head"
    binned = cut(
        series, bins=bins, right=False, include_lowest=True, labels=range(bins)
    )
    df = DataFrame(binned, dtype=float)

    if freq in ("M", "ME"):
        df["time"] = df.index.isocalendar().month
    elif freq == "W":
        df["time"] = df.index.isocalendar().week
    elif freq == "D":
        df["time"] = df.index.isocalendar().day
    else:
        msg = "freq %s is not a supported option."
        logger.error(msg, freq)
        raise ValueError(msg % freq)

    df["values"] = 1.0
    df = df.pivot_table(columns="head", index="time", aggfunc="sum", values="values")

    # Count of rows and column items
    x = df.sum(axis=1)  # Time
    y = df.sum(axis=0)  # Head
    z = series.size  # Total number of observations

    hx = -(x / z * log(x / z)).sum()
    hy = -(y / z * log(y / z)).sum()
    hxy = -(df / z * log(df / z, where=df.values != 0)).sum().sum()

    # Compute final components
    p = 1 - (hxy - hx) / log(bins)  # Predictability
    c = 1 - hy / log(bins)  # Constancy
    m = (hx + hy - hxy) / log(bins)  # Contingency
    return p, c, m


def colwell_constancy(
    series: Series,
    bins: int = 11,
    freq: str = "W",
    method: str = "mean",
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """Colwells constancy index after :cite:t:`colwell_predictability_1974`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int
        number of bins to determine the states of the groundwater.
    freq: str, optional
        frequency to resample the series to.
    method: str, optional
        Method to use for resampling. Only "mean" is allowed now.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    c: float
        Colwell's constancy.

    Notes
    -----
    One minus the sum of entropy with respect to state, divided by the logarithm of
    the absolute number of possible states.

    """
    return _colwell_components(
        series=series, bins=bins, freq=freq, method=method, normalize=normalize
    )[1]


def colwell_contingency(
    series: Series,
    bins: int = 11,
    freq: str = "W",
    method: str = "mean",
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """Colwell's contingency :cite:t:`colwell_predictability_1974`

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int
        number of bins to determine the states of the groundwater.
    freq: str, optional
        frequency to resample the series to.
    method: str, optional
        Method to use for resampling. Only "mean" is allowed now.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    m: float
         Colwell's contingency.

    Notes
    -----
    The difference between the sum of entropy for each time step and possible state
    of the seasonal cycle, and the overall entropy across all states and time steps,
    divided by the logarithm of the absolute number of possible states. Entropy
    according to definition in information theory, see reference for details.

    """
    return _colwell_components(
        series=series, bins=bins, freq=freq, method=method, normalize=normalize
    )[2]


def low_pulse_count(
    series: Series, quantile: float = 0.2, rolling_window: Union[str, None] = "7D"
) -> float:
    """Average number of times the series exceeds a certain threshold per year.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Average number of times the series exceeds a certain threshold per year.

    Notes
    -----
    Number of times during which the head drops below a certain threshold.
    The threshold is defined as the 20th percentile of non-exceedance
    :cite:t:`richter_method_1996`.

    Warning
    -------
    This method is sensitive to measurement noise, e.g., every change is sign in the
    differences is counted as a pulse. Therefore, it is recommended to smooth the time
    series first (which is also the default).

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    h = series < series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index

    # Deal with pulses in the beginning and end of the time series
    if h.iat[0]:
        sel = sel.append(series.index[:1]).sort_values()
    if h.iat[-1]:
        sel = sel.append(series.index[-1:]).sort_values()

    return sel.size / 2 / series.index.year.unique().size


def high_pulse_count(
    series: Series, quantile: float = 0.8, rolling_window: Union[str, None] = "7D"
) -> float:
    """Average number of times the series exceeds a certain threshold per year.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Average number of times the series exceeds a certain threshold per year.

    Notes
    -----
    Number of times during which the head exceeds a certain threshold. The threshold is
    defined as the 80th percentile of non-exceedance.

    Warning
    -------
    This method is sensitive to measurement noise, e.g., every change is sign in the
    differences is counted as a pulse. Therefore, it is recommended to smooth the time
    series first (which is also the default).

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    h = series > series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index
    if h.iat[0]:
        sel = sel.append(series.index[:1]).sort_values()
    if h.iat[-1]:
        sel = sel.append(series.index[-1:]).sort_values()
    return sel.size / 2 / series.index.year.unique().size


def low_pulse_duration(
    series: Series, quantile: float = 0.2, rolling_window: Union[str, None] = "7D"
) -> float:
    """Average duration of pulses where the head is below a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Average duration (in days) of pulses where the head drops below a certain
        threshold.

    Notes
    -----
    Average duration of pulses (in days) where the head drops below a certain threshold.

    Warning
    -------
    This method is sensitive to measurement noise, e.g., every change is sign in the
    differences is counted as a pulse. Therefore, it is recommended to smooth the time
    series first (which is also the default).

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    h = series < series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index

    if h.iat[0]:
        sel = sel.append(series.index[:1]).sort_values()
    if h.iat[-1]:
        sel = sel.append(series.index[-1:]).sort_values()

    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def high_pulse_duration(
    series: Series, quantile: float = 0.8, rolling_window: Union[str, None] = "7D"
) -> float:
    """Average duration of pulses where the head exceeds a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Average duration (in days) of pulses where the head drops below a certain
        threshold.

    Notes
    -----
    Average duration of pulses where the head drops exceeds a certain threshold. The
    threshold is defined as the 80th percentile of non-exceedance.

    Warning
    -------
    This method is sensitive to measurement noise, e.g., every change is sign in the
    differences is counted as a pulse. Therefore, it is recommended to smooth the time
    series first (which is also the default).

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    h = series > series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index

    if h.iat[0]:
        sel = sel.append(series.index[:1]).sort_values()
    if h.iat[-1]:
        sel = sel.append(series.index[-1:]).sort_values()

    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def _get_differences(series: Series, normalize: bool = False) -> Series:
    """Get the changes in the time series.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    differences: pandas.Series
        Differences in the time series in L/day.

    Notes
    -----
    Get the differences in the time series, and divide by the time step to get the rate
    of change. If normalize is True, the time series is normalized to values between
    zero and one.

    """
    if normalize:
        series = _normalize(series)

    dt = diff(series.index.to_numpy()) / Timedelta("1D")
    differences = series.diff().iloc[1:] / dt
    return differences


def rise_rate(
    series: Series, normalize: bool = False, rolling_window: Union[str, None] = "7D"
) -> float:
    """Mean of positive head changes from one day to the next.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Mean of positive head changes from one day to the next. The units of the rise
        rate are L/day (L defined by the input).

    Notes
    -----
    Mean rate of positive changes in head from one day to the next.

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    differences = _get_differences(series, normalize=normalize)
    rises = differences[differences > 0]

    return rises.mean()


def fall_rate(
    series: Series, normalize: bool = False, rolling_window: Union[str, None] = "7D"
) -> float:
    """Mean negative head changes from one day to the next.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Mean of negative head changes from one day to the next. The units of the fall
        rate are L/day (L defined by the input).

    Notes
    -----
    Mean rate of negative changes in head from one day to the next, according to
    :cite:t:`richter_method_1996`.

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    differences = _get_differences(series, normalize=normalize)
    falls = differences.loc[differences < 0]

    return falls.mean()


def cv_rise_rate(
    series: Series, normalize: bool = True, rolling_window: Union[str, None] = "7D"
) -> float:
    """Coefficient of Variation in rise rate.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Coefficient of Variation in rise rate.

    Notes
    -----
    Coefficient of variation in rise rate :cite:p:`richter_method_1996`. The higher the
    coefficient of variation, the more variable the rise rate is, and vice versa.

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    differences = _get_differences(series, normalize=normalize)
    rises = differences[differences > 0]

    return rises.std(ddof=1) / rises.mean()


def cv_fall_rate(
    series: Series, normalize: bool = False, rolling_window: Union[str, None] = "7D"
) -> float:
    """Coefficient of Variation in fall rate.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    rolling_window: str, optional
        Rolling window to use for smoothing the time series. Default is 7 days. Set to
        None to disable. See the pandas documentation for more information.

    Returns
    -------
    float:
        Coefficient of Variation in fall rate.

    Notes
    -----
    Coefficient of Variation in fall rate :cite:p:`richter_method_1996`. The higher the
    coefficient of variation, the more variable the fall rate is, and vice versa.

    """
    if rolling_window is not None:
        series = series.rolling(rolling_window).mean()

    differences = _get_differences(series, normalize=normalize)
    falls = differences[differences < 0]

    return falls.std(ddof=1) / falls.mean()


def magnitude(series: Series) -> float:
    """Difference between the minimum and maximum heads, divided by the minimum head
    adapted after :cite:t:`hannah_approach_2000`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float:
        Difference between the minimum and maximum heads, divided by the minimum head.

    Notes
    -----
    Difference between the minimum and maximum heads, divided by the minimum head:

    ..math::
        (h_max - h_min ) / h_min

    The higher the magnitude, the more variable the head is, and vice versa.

    """

    return (series.max() - series.min()) / series.min()


def reversals_avg(series: Series) -> float:
    """Average annual number of rises and falls in daily head.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float:
        Average number of rises and falls in daily head per year.

    Notes
    -----
    Average annual number of rises and falls (i.e., change of sign) in daily head
    :cite:p:`richter_method_1996`. The higher the number of reversals, the more
    variable the head is, and vice versa. If the head data is not daily, a warning is
    issued and nan is returned.

    """
    # Get the time step in days
    dt = diff(series.index.to_numpy()) / Timedelta("1D")

    # Check if the time step is approximately daily
    if not (dt > 0.9).all() & (dt < 1.1).all():
        msg = (
            "The time step is not approximately daily (>10%% of time steps are"
            "non-daily). This may lead to incorrect results."
        )
        logger.warning(msg)
        return nan
    else:
        series_diff = series.diff()
        reversals = (
            (series_diff[series_diff != 0.0] > 0).astype(int).diff().replace(-1, 1)
        )
        return reversals.resample(year_offset).sum().mean()


def reversals_cv(series: Series) -> float:
    """Coefficient of Variation in annual number of rises and falls.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float:
        Coefficient of Variation in annual number of rises and falls.

    Notes
    -----
    Coefficient of Variation in annual number of rises and falls in daily head
    :cite:p:`richter_method_1996`. If the coefficient of variation is high, the number
    of reversals is highly variable, and vice versa. If the head data is not daily, a
    warning is issued and nan is returned.

    """
    # Get the time step in days
    dt = diff(series.index.to_numpy()) / Timedelta("1D")

    # Check if the time step is approximately daily
    if not (dt > 0.9).all() & (dt < 1.1).all():
        msg = (
            "The time step is not approximately daily. "
            "This may lead to incorrect results."
        )
        logger.warning(msg)
        return nan
    else:
        series_diff = series.diff()
        reversals = (
            (series_diff[series_diff != 0.0] > 0).astype(int).diff().replace(-1, 1)
        )
        annual_sum = reversals.resample(year_offset).sum()
        return annual_sum.std(ddof=1) / annual_sum.mean()


def mean_annual_maximum(series: Series, normalize: bool = True) -> float:
    """Mean of annual maximum head after :cite:t:`clausen_flow_2000`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Mean of annual maximum head.

    Notes
    -----
    Mean of annual maximum head :cite:p:`clausen_flow_2000`.

    Warning
    -------
    This signatures is sensitive to the base level of the time series if normalize is
    set to False.

    """
    if normalize:
        series = _normalize(series)

    return series.resample(year_offset).max().mean()


def bimodality_coefficient(series: Series, normalize: bool = True) -> float:
    """Bimodality coefficient after :cite:t:`ellison_effect_1987`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Bimodality coefficient.

    Notes
    -----
    Squared product moment skewness (s) plus one, divided by product moment kurtosis
    (k):

    ..math::
        b = (s^2 + 1 ) / k

    Adapted from the R 'modes' package. The higher the bimodality coefficient, the more
    bimodal the head distribution is, and vice versa.

    """
    if normalize:
        series = _normalize(series)
    series = series.dropna()
    n = series.size
    series_mean_diff = series - series.mean()

    # Compute the skew for a finite sample
    skew = (
        (1 / n)
        * sum(series_mean_diff**3)
        / (((1 / n) * sum(series_mean_diff**2)) ** 1.5)
    )
    skew *= (sqrt(n * (n - 1))) / (n - 2)

    # Compute the kurtosis for a finite sample
    kurt = (1 / n) * sum(series_mean_diff**4) / (
        ((1 / n) * sum(series_mean_diff**2)) ** 2
    ) - 3
    kurt = ((n - 1) * ((n + 1) * kurt - 3 * (n - 1)) / ((n - 2) * (n - 3))) + 3

    return ((skew**2) + 1) / (kurt + ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))))


def _get_events_binned(
    series: Series,
    normalize: bool = False,
    up: bool = True,
    bins: int = 300,
    min_event_length: int = 10,
    min_n_events: int = 2,
) -> Series:
    """Get the recession or recovery events and bin them.

    Parameters
    ----------
    series : Series
        Pandas Series with DatetimeIndex and head values.
    normalize : bool, optional
        normalize the time series to values between zero and one.
    up : bool, optional
        If True, get the recovery events, if False, get the recession events.
    bins : int, optional
        Number of bins to bin the data to.
    min_event_length : int, optional
        Minimum length of an event in days. Events shorter than this are discarded.
    min_n_events : int, optional
        Minimum number of events in a bin. Bins with less events are discarded.

    Returns
    -------
    Series:
        Binned events.

    """
    if normalize:
        series = _normalize(series)

    series.name = "difference"  # Name the series for the split function

    # Get the negative differences
    h = series.dropna().copy()

    # Set the negative differences to nan if up is True, and the positive differences
    # to nan if up is False (down).
    if up:
        h[h.diff() < 0] = nan
    else:
        h[h.diff() > 0] = nan

    # Split the data into events
    events = split(h, where(isnan(h.values))[0])
    events = [ev[~isnan(ev.values)] for ev in events if not isinstance(ev, ndarray)]

    events_new = []

    for ev in events:
        # Drop empty events and events shorter than min_events_length.
        if ev.empty or ev.index.size < 2:
            pass
        else:
            ev.index = (ev.index - ev.index[0]).days
            if ev.index[-1] > min_event_length:
                events_new.append(ev)

    if len(events_new) == 0:
        return Series(dtype=float)
    events = concat(events_new, axis=1)

    # Subtract the absolute value of the first day of each event
    data = events - events.iloc[0, :]
    data = data.loc[:, data.sum() != 0.0]  # Drop columns with only zeros (no events)

    # Bin the data and compute the mean
    binned = Series(dtype=float)
    for g in data.groupby(
        cut(data.index, bins=min(bins, data.index.max())), observed=False
    ):
        # Only use bins with more than 5 events
        if g[1].dropna(axis=1).columns.size > min_n_events:
            value = g[1].dropna(axis=1).mean(axis=1)
            if not value.empty:
                binned[g[0].mid] = value.iat[0]

    binned = binned[binned != 0].dropna()
    return binned


def recession_constant(
    series: Series,
    bins: int = 300,
    normalize: bool = False,
    min_event_length: int = 10,
    min_n_events: int = 2,
) -> float:
    """Recession constant adapted after :cite:t:`kirchner_catchments_2009`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int, optional
        Number of bins to bin the data to.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    min_event_length: int, optional
        Minimum length of an event in days. Events shorter than this are discarded.
    min_n_events: int, optional
        Minimum number of events in a bin. Bins with less events are discarded.

    Returns
    -------
    float:
        Recession constant in days.

    Notes
    -----
    Recession constant adapted after :cite:t:`kirchner_catchments_2009`, which is the
    decay constant of the exponential model fitted to the percentile-wise binned means
    of the recession segments. The higher the recession constant, the slower the head
    declines, and vice versa. The following function is fitted to the binned data
    (similar to the Exponential response function in Pastas):

    ..math::
        h(t) = - h_0 * (1 - exp(-t / c))

    where h(t) is the head at time t, h_0 is the final head as t goes to infinity, and
    c is the recession constant.

    """
    binned = _get_events_binned(
        series,
        normalize=normalize,
        up=False,
        bins=bins,
        min_event_length=min_event_length,
        min_n_events=min_n_events,
    )

    # Deal with the case that binned is empty
    if binned.empty:
        return nan

    # Fit an exponential model to the binned data and return the decay constant
    def f(t, *p):
        return -p[0] * (1 - exp(-t / p[1]))

    popt, _ = curve_fit(
        f, binned.index, binned.values, p0=[1, 100], bounds=(0, [100, 1e3])
    )

    # Return nan and raise warning if the decay constant is close to the boundary
    if isclose(popt[1], 0.0) or isclose(popt[1], 1e3):
        msg = (
            "The estimated recession constant (%s) is close to the boundary. "
            "This may lead to incorrect results."
        )
        logger.warning(msg, round(popt[1], 2))
        return nan
    else:
        return popt[1]


def recovery_constant(
    series: Series,
    bins: int = 300,
    normalize: bool = False,
    min_event_length: int = 10,
    min_n_events: int = 2,
) -> float:
    """Recovery constant after :cite:t:`kirchner_catchments_2009`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int, optional
        Number of bins to bin the data to.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    min_event_length: int, optional
        Minimum length of an event in days. Events shorter than this are discarded.
    min_n_events: int, optional
        Minimum number of events in a bin. Bins with less events are discarded.

    Returns
    -------
    float:
        Recovery constant.

    Notes
    -----
    Time constant of the exponential function fitted to percentile-wise binned means
    of the recovery segments. The higher the recovery constant, the slower the head
    recovers, and vice versa. The following function is fitted to the binned data
    (similar to the Exponential response function in Pastas):

    ..math::
        h(t) = h_0 * (1 - exp(-t / c))

    where h(t) is the head at time t, h_0 is the final head as t goes to infinity, and
    c is the recovery constant.

    """
    binned = _get_events_binned(
        series,
        normalize=normalize,
        up=True,
        bins=bins,
        min_event_length=min_event_length,
    )

    # Deal with the case that binned is empty
    if binned.empty:
        return nan

    # Fit an exponential model to the binned data and return the decay constant
    def f(t, *p):
        return -p[0] * (1 - exp(-t / p[1]))

    popt, _ = curve_fit(
        f, binned.index, binned.values, p0=[1, 100], bounds=(0, [100, 1e3])
    )

    # Return nan and raise warning if the recovery constant is close to the boundary
    if isclose(popt[1], 0.0) or isclose(popt[1], 1e3):
        msg = (
            "The estimated recovery constant (%s) is close to the boundary. "
            "This may lead to incorrect results."
        )
        logger.warning(msg, round(popt[1], 2))
        return nan
    else:
        return popt[1]


def duration_curve_slope(
    series: Series,
    l: float = 0.1,  # noqa: E741
    u: float = 0.9,
    normalize: bool = False,
) -> float:
    """Slope of the head duration curve between percentile l and u after
    :cite:t:`oudin_are_2010`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    l: float, optional
        lower percentile, a float between 0 and 1, lower than u.
    u: float, optional
        upper percentile, a float between 0 and 1, higher than l.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Slope of the head duration curve between percentile l and u.

    Notes
    -----
    Slope of the head duration curve between percentile l and u. The more negative the
    slope, the more values are above or below the percentile l and u, and vice versa.

    Note that the slope is negative, contrary to the flow duration curve commonly used
    in surface water hydrology.

    """
    if normalize:
        series = _normalize(series)

    # Get the series between the percentiles
    s = series[
        (series > series.quantile(l)) & (series < series.quantile(u))
    ].sort_values(ascending=False)

    # Deal with the case that s is empty
    if s.empty:
        return nan

    s.index = linspace(0, 1, s.size)
    return linregress(s.index, s.values).slope


def duration_curve_ratio(
    series: Series,
    l: float = 0.1,  # noqa: E741
    u: float = 0.9,
    normalize: bool = True,
) -> float:
    """Ratio of the head duration curve between the percentile l and u after
    :cite:t:`richards_measures_1990`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    l: float
        lower percentile, a float between 0 and 1, lower than u.
    u: float, optional
        upper percentile, a float between 0 and 1, higher than l.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        Ratio of the duration curve between the percentile l and u.

    Notes
    -----
    Ratio of the duration curve between the percentile l and u. The higher the ratio,
    the flatter the head duration curve, and vice versa.

    """
    if normalize:
        series = _normalize(series)

    return series.quantile(l) / series.quantile(u)


def richards_pathlength(series: Series, normalize: bool = True) -> float:
    """The path length of the time series, standardized by time series length after
    :cite:t:`baker_new_2004`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float:
        The path length of the time series, standardized by time series length and
        median.

    Notes
    -----
    The path length of the time series, standardized by time series length and median.

    """
    if normalize:
        series = _normalize(series)

    series = series.dropna()
    dt = diff(series.index.to_numpy()) / Timedelta("1D")
    dh = series.diff().dropna()

    # sum(dt) is more fair with irregular time series
    return sum(sqrt(dh**2 + dt**2)) / (sum(dt) * series.median())


def _baselevel(
    series: Series, normalize: bool = True, period="30D"
) -> Tuple[Series, Series]:
    """Baselevel function for the baselevel index and stability.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    period: str, optional
        Period to resample the time series to in days (e.g., '10D' or '90D'). Default
        is 30 days.

    Returns
    -------
    series: pandas.Series
        Pandas Series of the original for
    ht: pandas.Series
        Pandas Series for the base head

    """
    if normalize:
        series = _normalize(series)

    # A/B. Selecting minima hm over a period
    hm = series.resample(period).min().dropna()
    series = series.resample("D").interpolate()

    # C. define the turning point ht (0.9 * head < adjacent heads)
    ht = Series(index=[hm.index[0]], data=[hm.iat[0]], dtype=float)

    for i, h in enumerate(hm.iloc[1:-1], start=1):
        if (0.9 * h < hm.iat[i - 1]) & (0.9 * h < hm.iat[i + 1]):
            ht[hm.index[i]] = h

    ht[hm.index[-1]] = hm.iat[-1]

    # ensure that index is a DatetimeIndex
    ht.index = to_datetime(ht.index)

    # D. Interpolate
    ht = ht.resample("D").interpolate()

    # E. Assign a base head to each day
    ht[ht > series.loc[ht.index]] = series

    return series, ht


def baselevel_index(series: Series, normalize: bool = True, period="30D") -> float:
    """Base level index (BLI) adapted after :cite:t:`organization_manual_2008`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    period: str, optional
        Period to resample the time series to in days (e.g., '10D' or '90D'). Default
        is 30 days.

    Returns
    -------
    float:
        Base level index.

    Notes
    -----
    Adapted analogously to its application in streamflow. Here, a baselevel time
    series is separated from a X-day minimum head in a moving window. BLI
    equals the total sum of heads of original time series divided by the total sum of
    heads from the baselevel time series. Both these time series are resampled to daily
    heads by interpolation for consistency.

    """

    series, ht = _baselevel(series, normalize=normalize, period=period)
    return ht.sum() / series.sum()


def baselevel_stability(series: Series, normalize: bool = True, period="30D") -> float:
    """Baselevel stability after :cite:t:`heudorfer_index-based_2019`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.
    period: str, optional
        Period to resample the time series to, in days (e.g., '10D' or '90D'). Default
        is 30 days.

    Returns
    -------
    float:
        Base level stability.

    Notes
    -----
    Originally developed for streamflow, here the Base Flow Index algorithm is
    analogously adapted to groundwater time series as a filter to separate the slow
    component (“base level") of the time series. Then, the mean annual base level is
    calculated. Base Level Stability is the difference of maximum and minimum annual
    base level.

    """

    _, ht = _baselevel(series, normalize=normalize, period=period)

    return ht.resample(year_offset).mean().max() - ht.resample(year_offset).mean().min()


def autocorr_time(series: Series, cutoff: float = 0.8, **kwargs) -> float:
    """Lag where the autocorrelation function exceeds a cut-off value.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    cutoff: float, optional
        Cut-off value for the autocorrelation function. Default is 0.7.
    kwargs: dict, optional
        Additional keyword arguments are passed to the pastas acf method.

    Returns
    -------
    float:
        Lag in days where the autocorrelation function exceeds the cutoff value.

    Notes
    -----
    Lag in days where the autocorrelation function exceeds the cutoff value for the
    first time. The higher the lag, the more autocorrelated the time series is, and
    vice versa. In practical terms higher values mean that the groundwater system has
    a longer memory and the response to changes in the forcing are visible longer in
    the head time series.

    """
    c = acf(series.dropna(), **kwargs)  # Compute the autocorrelation function

    if c.min() > cutoff:
        return nan
    else:
        return (c < cutoff).idxmax() / Timedelta("1D")


def _date_min_max(series: Series, stat: str) -> float:
    """Compute the average date of the minimum head value with circular statistics.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    stat: str
        Either "min" or "max". If "min", the average date of the minimum head value is
        computed. If "max", the average date of the maximum head value is computed.

    Returns
    -------
    float:
        Average date of the minimum or maximum head value.

    Notes
    -----
    The average date is computed by taking the average of the day of the year of the
    minimum head value for each year, using circular statistics. We refer to
    :cite:t:`fisher_statistical_1995` (page 31) for more information on circular
    statistics.

    """
    # Get the day of the year of the minimum head value for each year
    if stat == "min":
        data = series.groupby(series.index.year).idxmin(skipna=True).dropna().values
    elif stat == "max":
        data = series.groupby(series.index.year).idxmax(skipna=True).dropna().values

    doy = DatetimeIndex(data).dayofyear.to_numpy(float)

    m = 365.25
    two_pi = 2 * pi

    thetas = array(doy) * two_pi / m
    c = cos(thetas).sum()
    s = sin(thetas).sum()

    if (s > 0) & (c > 0):
        mean_theta = arctan(s / c)
    elif c < 0:
        mean_theta = arctan(s / c) + pi
    elif (s < 0) & (c > 0):
        mean_theta = arctan(s / c) + two_pi
    else:
        # This should never happen
        raise ValueError("Something went wrong in the circular statistics.")

    return mean_theta * 365.25 / two_pi


def date_min(series: Series) -> float:
    """Compute the average date of the minimum head value with circular statistics.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float:
        Average date of the minimum head value.

    Notes
    -----
    Average date of the minimum head value. The higher the date, the later the minimum
    head value occurs in the year, and vice versa.

    The average date is computed by taking the average of the day of the year of the
    minimum head value for each year, using circular statistics. We refer to
    :cite:t:`fisher_statistical_1995` (page 31) for more information on circular
    statistics.

    """
    return _date_min_max(series, "min")


def date_max(series: Series) -> float:
    """Compute the average date of the maximum head value with circular statistics.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float:
        Average date of the maximum head value.

    Notes
    -----
    Average date of the maximum head value. The higher the date, the later the maximum
    head value occurs in the year, and vice versa.

    The average date is computed by taking the average of the day of the year of the
    maximum head value for each year, using circular statistics. We refer to
    :cite:t:`fisher_statistical_1995` (page 31) for more information on circular
    statistics.

    """
    return _date_min_max(series, "max")


def summary(
    data: Union[DataFrame, Series], signatures: Optional[list] = None
) -> DataFrame:
    """Method to get many signatures for a time series.

    Parameters
    ----------
    data: Union[pandas.DataFrame, pandas.Series]
        pandas DataFrame or Series with DatetimeIndex
    signatures: list
        list of signatures to return. By default all available signatures are returned.

    Returns
    -------
    result: pandas.DataFrame
        Pandas DataFrame with every row a signature and the signature value for each column.

    Examples
    --------
    >>> idx = date_range("2000", "2010")
    >>> data = np.random.rand(len(idx), 3)
    >>> df = DataFrame(index=idx, data=data, columns=[year_offset, "B", "C"], dtype=float)
    >>> ps.stats.signatures.summary(df)

    """
    if signatures is None:
        signatures = __all__

    if isinstance(data, DataFrame):
        result = DataFrame(index=signatures, columns=data.columns, dtype=float)
    elif isinstance(data, Series):
        result = DataFrame(index=signatures, columns=[data.name], dtype=float)
    else:
        raise ValueError("Invalid data type. Expected DataFrame or Series.")

    # Get the signatures
    for signature in signatures:
        # Check if the signature is valid
        if signature not in __all__:
            msg = "Signature %s is not a valid signature."
            logger.error(msg, signature)
            raise ValueError(msg % signature)

        # Get the function and compute the signature for each column/series
        func = getattr(ps.stats.signatures, signature)
        if isinstance(data, DataFrame):
            result.loc[signature] = data.apply(func)
        elif isinstance(data, Series):
            result.loc[signature] = func(data)

    return result
