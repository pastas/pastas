"""This module contains methods to compute the groundwater signatures."""
# Type Hinting
from typing import Optional, Tuple

import pandas as pd
from numpy import arange, diff, log, nan, sqrt
from pandas import DatetimeIndex, Series, Timedelta, cut
from scipy.stats import linregress

import pastas as ps

__all__ = [
    "cv_period_mean",
    "cv_date_min",
    "cv_fall_rate",
    "cv_rise_rate",
    "parde_seasonality",
    "avg_seasonal_fluctuation",
    "magnitude",
    "interannual_variation",
    "low_pulse_count",
    "high_pulse_count",
    "low_pulse_duration",
    "high_pulse_duration",
    "amplitude_range",
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
    "duration_curve_range",
    "baseflow_index",
    "richards_pathlength",
    "richards_baker_index",
    "baseflow_stability",
]


def _normalize(series: Series) -> Series:
    series = (series - series.min()) / (series.max() - series.min())
    return series


def cv_period_mean(series: Series, freq: str = "M") -> float:
    """Coefficient of variation of mean head over a period (default monthly).

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    freq: str, optional
        frequency to resample the series to by averaging.

    Returns
    -------
    cv: float
        Coefficient of variation of mean head over a period (default monthly).

    Notes
    -----
    Coefficient of variation of mean monthly heads :cite:t:`hughes_hydrological_1989`.

    """
    series = series.resample(freq).mean()
    cv = series.std() / series.mean()
    return cv


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
    Coefficient of variation of the date of annual minimum groundwater head
    according to :cite:t:`richter_method_1996`.

    """
    data = series.groupby(series.index.year).idxmin().dropna().values
    data = DatetimeIndex(data).dayofyear.to_numpy(float)
    cv = data.std() / data.mean()
    return cv


def parde_seasonality(series: Series, normalize: bool = True) -> float:
    """Parde seasonality according to :cite:t:`parde_fleuves_1933`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------

    Notes
    -----
    Pardé seasonality is the difference between the maximum and minimum Pardé
    coefficient. A Pardé series consists of 12 Pardé coefficients, corresponding to
    12 months. Pardé coefficient for, for example, January is its long‐term monthly
    mean groundwater head divided by the overall mean groundwater head.

    """
    coefficients = parde_coefficients(series=series, normalize=normalize)
    return coefficients.max() - coefficients.min()


def parde_coefficients(series: Series, normalize: bool = True) -> float:
    """Parde coefficients for each month :cite:t:`parde_fleuves_1933`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------

    Notes
    -----
    Pardé seasonality is the difference between the maximum and minimum Pardé
    coefficient. A Pardé series consists of 12 Pardé coefficients, corresponding to
    12 months. Pardé coefficient for, for example, January is its long‐term monthly
    mean groundwater head divided by the overall mean groundwater head.

    """
    if normalize:
        series = _normalize(series)

    coefficients = series.groupby(series.index.month).mean() / series.mean()
    coefficients.index.name = "month"
    return coefficients


def _martens(series: Series, normalize: bool = True) -> Tuple[Series, Series]:
    """Functions for the average seasonal fluctuation and inter annual fluctuation.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    hl: pandas.Series
        Lowest heads
    hw: pandas.Series
        Largest heads

    Notes
    -----
    According to :cite:t:`martens_groundwater_2013`.

    """

    if normalize:
        series = _normalize(series)

    s = series.resample("M")
    hl = s.min().groupby(s.min().index.year).nsmallest(3).groupby(level=0).mean()
    hw = s.max().groupby(s.max().index.year).nlargest(3).groupby(level=0).mean()

    return hl, hw


def avg_seasonal_fluctuation(series: Series, normalize: bool = True) -> float:
    """Classification according to :cite:t:`martens_groundwater_2013`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------

    float

    Notes
    -----
    Mean annual difference between the averaged 3 highest monthly groundwater heads
    per year and the averaged 3 lowest monthly groundwater heads per year.

    Average seasonal fluctuation (s):

        s = MHW - MLW

    """

    hl, hw = _martens(series, normalize=normalize)

    return hw.mean() - hl.mean()


def interannual_variation(series: Series, normalize: bool = True) -> float:
    """Interannual variation after :cite:t:`martens_groundwater_2013`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    The average between the range in annually averaged 3 highest monthly groundwater
    heads and the range in annually averaged 3 lowest monthly groundwater heads.

    Inter-yearly variation of high and low water table (y):

        y = ((max_HW - min_HW) + (max_LW - min_LW)) / 2

    Warning: In this formulating the water table is references to a certain datum and
    positive, not as depth below the surface!

    """

    hl, hw = _martens(series, normalize=normalize)

    return (hw.max() - hw.min()) + (hl.max() - hl.min()) / 2


def colwell_components(
    series: Series,
    bins: int = 11,
    freq: str = "M",
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
        frequency to resample the series to.
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
    df = pd.DataFrame(binned)
    df["time"] = df.index.month
    df["values"] = 1
    df = df.pivot_table(columns="head", index="time", aggfunc="sum", values="values")

    # Count of rows and column items
    x = df.sum(axis=1)  # Time
    y = df.sum(axis=0)  # Head
    z = series.size  # Total number of observations

    hx = -(x / z * log(x / z)).sum()
    hy = -(y / z * log(y / z)).sum()
    hxy = -(df / z * log(df / z, where=df != 0)).sum().sum()

    # Compute final components
    p = 1 - (hxy - hy) / log(bins)  # Predictability
    c = 1 - hx / log(bins)  # Constancy
    m = (hx + hy - hxy) / log(bins)  # Contingency
    return p, c, m


def colwell_constancy(
    series: Series,
    bins: int = 11,
    freq: str = "M",
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
    p, c, m: float, float, float
        predictability, constancy, contingency

    Notes
    -----
    One minus the sum of entropy with respect to state, divided by the logarithm of
    the absolute number of possible states.

    """
    return colwell_components(
        series=series, bins=bins, freq=freq, method=method, normalize=normalize
    )[1]


def colwell_contingency(
    series: Series,
    bins: int = 11,
    freq: str = "M",
    method: str = "mean",
    normalize: bool = True,
) -> Tuple[float, float, float]:
    """Colwell contingency :cite:t:`colwell_predictability_1974`

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
    p, c, m: float, float, float
        predictability, constancy, contingency

    Notes
    -----
    The difference between the sum of entropy for each time step and possible state
    of the seasonal cycle, and the overall entropy across all states and time steps,
    divided by the logarithm of the absolute number of possible states. Entropy
    according to definition in information theory, see reference for details.

    """
    return colwell_components(
        series=series, bins=bins, freq=freq, method=method, normalize=normalize
    )[2]


def low_pulse_count(series: Series, quantile: float = 0.2) -> int:
    """Number of times the series drops below a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------
    int:
        Number of times the series exceeds a certain threshold.

    Notes
    -----
    Number of times during which the groundwater head drops below a certain threshold.
    The threshold is defined as the 20th percentile of non-exceedance
    :cite:t:`richter_method_1996`.

    """
    h = series < series.quantile(quantile)
    return (h.astype(int).diff() > 0).sum()


def high_pulse_count(series: Series, quantile: float = 0.8) -> int:
    """Number of times the series exceeds a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------
    h: int
        Number of times the series exceeds a certain threshold.

    Notes
    -----
    Number of times during which the groundwater head exceeds a certain threshold.
    The threshold is defined as the 80th percentile of non-exceedance.

    """
    h = series > series.quantile(quantile)
    return (h.astype(int).diff() > 0).sum()


def low_pulse_duration(series: Series, quantile: float = 0.8) -> float:
    """Average duration of pulses where the head is below a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------
    float

    Notes
    -----
    Average duration of pulses where the groundwater head drops below a certain
    threshold. The threshold is defined as the 20th percentile of non-exceedance.

    """
    h = series < series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index
    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def high_pulse_duration(series: Series, quantile: float = 0.8) -> float:
    """Average duration of pulses where the head exceeds a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------
    float

    Notes
    -----
    Average duration of pulses where the groundwater head drops exceeds a certain
    threshold. The threshold is defined as the 80th percentile of non-exceedance.

    """
    h = series > series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, nan).shift(-1).dropna().index
    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def amplitude_range(series: Series) -> float:
    """Range of unscaled groundwater head.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float

    Notes
    -----
    Range of unscaled groundwater head.
    """
    return series.max() - series.min()


def rise_rate(series: Series, normalize: bool = False) -> float:
    """Mean of positive head changes from one day to the next.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Mean rate of positive changes in head from one day to the next.

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    rises = difference[difference > 0]
    return rises.mean()


def fall_rate(series: Series, normalize: bool = False) -> float:
    """Mean negative head changes from one day to the next.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Mean rate of negative changes in head from one day to the next, according to
    :cite:t:`richter_method_1996`.

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    falls = difference[difference < 0]
    return falls.mean()


def cv_rise_rate(series: Series, normalize: bool = False) -> float:
    """Coefficient of Variation in rise rate.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Coefficient of Variation in rise rate :cite:p:`richter_method_1996`.

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    rises = difference[difference > 0]
    return rises.std() / rises.mean()


def cv_fall_rate(series: Series, normalize: bool = False) -> float:
    """Coefficient of Variation in fall rate.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Coefficient of Variation in fall rate :cite:p:`richter_method_1996`.

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    falls = difference[difference < 0]
    return falls.std() / falls.mean()


def magnitude(series: Series) -> float:
    """Difference of peak head to base head, divided by base head after
    :cite:t:`hannah_approach_2000`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float

    Notes
    -----
    Difference of peak head to base head, divided by base head.

      ..math:: (h_max - h_min ) / h_min

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
    float

    Notes
    -----
    Average annual number of rises and falls (i.e., change of sign) in daily head
    :cite:p:`richter_method_1996`.

    """
    reversals = (series.diff() > 0).astype(int).diff().replace(-1, 1)
    return reversals.resample("A").sum().mean()


def reversals_cv(series: Series) -> float:
    """Coefficient of Variation in annual number of rises and falls.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.

    Returns
    -------
    float

    Notes
    -----
    Coefficient of Variation in annual number of rises and falls in daily head
    :cite:p:`richter_method_1996`.

    """
    reversals = (
        (series.diff() > 0).astype(int).diff().replace(-1, 1).resample("A").sum()
    )
    return reversals.std() / reversals.mean()


def mean_annual_maximum(series: Series, normalize: bool = False) -> float:
    """Mean of annual maximum after :cite:t:`clausen_flow_2000`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    """
    if normalize:
        series = _normalize(series)

    return series.resample("A").max().mean()


def bimodality_coefficient(series: Series, normalize: bool = False) -> float:
    """Bimodality coefficient after :cite:t:`ellison_effect_1987`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Squared product moment skewness (s) plus one, divided by product moment kurtosis
    (k).

    ..math:: b = (s^2 + 1 ) / k

    Adapted from the R 'modes' package.

    """
    if normalize:
        series = _normalize(series)
    series = series.dropna()
    n = series.size
    # Compute the skew for a finite sample
    skew = (
        (1 / n)
        * sum((series - series.mean()) ** 3)
        / (((1 / n) * sum((series - series.mean()) ** 2)) ** 1.5)
    )
    skew *= (sqrt(n * (n - 1))) / (n - 2)

    # Compute the kurtosis for a finite sample
    kurt = (1 / n) * sum((series - series.mean()) ** 4) / (
        ((1 / n) * sum((series - series.mean()) ** 2)) ** 2
    ) - 3
    kurt = ((n - 1) * ((n + 1) * kurt - 3 * (n - 1)) / ((n - 2) * (n - 3))) + 3

    return ((skew**2) + 1) / (kurt + ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))))


def recession_constant(
    series: Series, bins: int = 20, normalize: bool = False
) -> float:
    """Recession constant after :cite:t:`kirchner_catchments_2009`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int, optional
        Number of bins to bin the data to.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Slope of the linear model fitted to percentile‐wise binned means in a log‐log
    plot of negative head versus negative head one time step ahead.

    """
    if normalize:
        series = _normalize(series)

    series.name = "diff"
    df = pd.DataFrame(series.diff().loc[series.diff() < 0], columns=["diff"])
    df["head"] = series.loc[df.index]

    binned = pd.Series(dtype=float)
    for g in df.groupby(pd.cut(df["head"], bins=bins)):
        binned[g[0].mid] = g[1]["diff"].mean()

    binned = binned.dropna()
    fit = linregress(log(binned.index), log(-binned.values))

    return fit.slope


def recovery_constant(series: Series, bins: int = 20, normalize: bool = False) -> float:
    """Recovery constant after :cite:t:`kirchner_catchments_2009`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    bins: int, optional
        Number of bins to bin the data to.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Slope of the linear model fitted to percentile‐wise binned means in a log‐log
    plot of positive head versus positive head one time step ahead.

    """
    if normalize:
        series = _normalize(series)

    series.name = "diff"
    df = pd.DataFrame(series.diff().loc[series.diff() > 0], columns=["diff"])
    df["head"] = series.loc[df.index]

    binned = pd.Series(dtype=float)
    for g in df.groupby(pd.cut(df["head"], bins=bins)):
        binned[g[0].mid] = g[1]["diff"].mean()

    binned = binned.dropna()
    fit = linregress(log(binned.index), log(binned.values))

    return fit.slope


def duration_curve_slope(
    series: Series, l: float = 0.1, u: float = 0.9, normalize: bool = True
) -> float:
    """Slope of the duration curve between percentile l and u after
    :cite:t:`oudin_are_2010`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    l: float
        lower percentile, a float between 0 and 1, lower than u
    u: float, optional
        upper percentile, a float between 0 and 1, higher than l.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Slope of the duration curve (analogue flow duration curve for streamflow) between
    percentile l and u.

    """
    if normalize:
        series = _normalize(series)

    s = series[
        (series.quantile(l) > series) & (series < series.quantile(u))
    ].sort_values()
    s.index = arange(s.size) / s.size
    return linregress(s.index, s.values).slope


def duration_curve_range(
    series: Series, l: float = 0.1, u: float = 0.9, normalize: bool = True
) -> float:
    """Range of the duration curve between the percentile l and u after
    :cite:t:`richards_measures_1990`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    l: float
        lower percentile, a float between 0 and 1, lower than u
    u: float, optional
        upper percentile, a float between 0 and 1, higher than l.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Range of the duration curve between the percentile l and u.

    """
    if normalize:
        series = _normalize(series)

    return series.quantile(u) - series.quantile(l)


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
    float

    Notes
    -----
    The path length of the time series, standardized by time series length and the
    median head.

    """
    if normalize:
        series = _normalize(series)

    series = series.dropna()
    dt = diff(series.index.to_numpy()) / Timedelta("1D")
    dh = series.diff().dropna()
    # sum(dt) is more fair with irregular time series
    return sum(sqrt(dh**2 + dt**2)) / sum(dt)


def richards_baker_index(series: Series, normalize: bool = True) -> float:
    """Richards-Baker index according to :cite:t:`baker_new_2004`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Sum of absolute values of day‐to‐day changes in head divided by the sum of scaled
    daily head. Equivalent the Richards Pathlength without the time component.

    """
    if normalize:
        series = _normalize(series)

    return series.diff().dropna().abs().sum() / series.sum()


def _baseflow(series: Series, normalize: bool = True) -> Tuple[Series, Series]:
    """Baseflow function for the baseflow index and stability.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    series: pandas.Series
        Pandas Series of the original for
    ht: pandas.Series
        Pandas Series for the base head
    """
    if normalize:
        series = _normalize(series)

    # A/B. Selecting minima hm over 5 day periods
    hm = series.resample("5D").min().dropna()

    # C. define the turning points ht (0.9 * head < adjacent heads)
    ht = pd.Series(dtype=float)
    for i, h in enumerate(hm.iloc[1:-1], start=1):
        if (h < hm.iloc[i - 1]) & (h < hm.iloc[i + 1]):
            ht[hm.index[i]] = h

    # ensure that index is a DatetimeIndex
    ht.index = pd.to_datetime(ht.index)

    # D. Interpolate
    ht = ht.resample("D").interpolate()

    # E. Assign a base head to each day
    ht[ht > series.resample("D").mean().loc[ht.index]] = series.resample("D").mean()

    return series, ht


def baseflow_index(series: Series, normalize: bool = True) -> float:
    """Baseflow index according to :cite:t:`organization_manual_2008`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Adapted analogously to its application in streamflow. Here, a baseflow time
    series is separated from a 5‐day minimum groundwater head in a moving window. BFI
    equals the total sum of heads of original time series divided by the total sum of
    heads from the baseflow type of time series.

    """

    series, ht = _baseflow(series, normalize=normalize)

    return series.resample("D").mean().sum() / ht.sum()


def baseflow_stability(series: Series, normalize: bool = True) -> float:
    """Baseflow stability after :cite:t:`heudorfer_index-based_2019`.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    normalize: bool, optional
        normalize the time series to values between zero and one.

    Returns
    -------
    float

    Notes
    -----
    Originally developed for streamflow, here the Base Flow Index algorithm is
    analogously adapted to groundwater time series as a filter to separate the slow
    component (“baseflow”) of the time series. Then, the mean annual baseflow is
    calculated. Base Flow Stability is the difference of maximum and minimum annual
    baseflow.

    """

    series, ht = _baseflow(series, normalize=normalize)

    return ht.resample("A").mean().max() - ht.resample("A").mean().min()


def hurst_exponent(series: Series):
    """Hurst exponent according to :cite:t:`wang_characteristic-based_2006`.

    Returns
    -------

    Notes
    -----
    The slope of a linear model fitted to the relationship between the sample size
    and the logarithmized sample range of k contiguous subsamples from the time series.

    """
    return NotImplementedError


def autocorr(series: Series, freq: str = "w"):
    """Lag where the first peak in the autocorrelation function occurs after
    :cite:t:`wang_characteristic-based_2006`.

    Returns
    -------

    Notes
    -----
    Lag where the first peak in the autocorrelation function occurs.

    """
    return NotImplementedError


def lyapunov_exponent(series: Series):
    """The exponential rate of divergence of nearby data points after
    :cite:t:`hilborn_chaos_2000`.

    Returns
    -------

    Notes
    -----
    The exponential rate of divergence of nearby data points when moving away in time
    from a certain data point in the series. Iteratively estimated for every point in
    the time series, then averaged.

    """
    return NotImplementedError


def peak_timescale(series: Series):
    """Area under peak divided by difference of peak head to peak base after
    :cite:t:`gaal_flood_2012`.

    Returns
    -------

    Notes
    -----
    Area under peak divided by difference of peak head to peak base, averaged over
    all peaks.

    """
    return NotImplementedError


def excess_mass(series: Series):
    """Test statistic of the dip test, after :cite:t:`hartigan_dip_1985`.

    Returns
    -------

    Notes
    -----
    Test statistic of the dip test; maximum distance between the empirical
    distribution and the best fitting unimodal distribution. By default, the best
    fitting distribution is the uniform.

    """
    return NotImplementedError


def critical_bandwidth(series: Series):
    """Test statistic of the Silverman test, after :cite:t:`silverman_using_1981`.

    Returns
    -------

    Notes
    -----
    Test statistic of the Silverman test; minimum kernel bandwidth required to create
    an unimodal distribution estimated by fitting a Kernel Density Estimation.

    """
    return NotImplementedError


def peak_base_time(series: Series):
    """Difference between peak and base head, standardized by duration of peak after
    :cite:t:`heudorfer_index-based_2019`.

    Returns
    -------

    Notes
    -----
    Difference between peak and base head, standardized by duration of peak.

    """
    return NotImplementedError


def summary(series: Series, signatures: Optional[list] = None) -> Series:
    """Method to get many signatures for a time series.

    Parameters
    ----------
    series: pandas.Series
        pandas Series with DatetimeIndex
    signatures: list
        By default all available signatures are returned.

    Returns
    -------
    data: pandas.Series
        Pandas series with every row a signature

    Examples
    --------
    >>> idx = pd.date_range("2000", "2010")
    >>> head = pd.Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    >>> ps.stats.signatures.summary(head)
    """
    if signatures is None:
        signatures = __all__

    data = pd.Series(index=signatures, dtype=float)

    for signature in signatures:
        func = getattr(ps.stats.signatures, signature)
        data.loc[signature] = func(series)

    return data
