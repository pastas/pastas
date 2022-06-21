"""This module contains methods to compute the groundwater signatures."""
import pandas as pd
from pandas import NA, Timedelta, DatetimeIndex, cut
from numpy import diff, sqrt, log, arange
import pastas as ps
from scipy.stats import linregress

__all__ = ["cv_period_mean", "cv_date_min", "cv_fall_rate", "cv_rise_rate",
           "parde_seasonality", "avg_seasonal_fluctuation", "magnitude",
           "interannual_variation", "low_pulse_count", "high_pulse_count",
           "low_pulse_duration", "high_pulse_duration", "amplitude_range",
           "bimodality_coefficient", "mean_annual_maximum", "rise_rate",
           "fall_rate", "reversals_avg", "reversals_cv", "colwell_contingency",
           "colwell_constancy", "recession_constant", "recovery_constant",
           "duration_curve_slope", "duration_curve_range", "baseflow_index",
           "richards_pathlength", "richards_baker_index", "baseflow_stability"]


def _normalize(series):
    series = (series - series.min()) / (series.max() - series.min())
    return series


def cv_period_mean(series, freq="M"):
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
    Coefficient of variation of mean monthly heads [hughes_1989]_.

    References
    ----------
    .. [hughes_1989] Hughes, J., & James, B. (1989). A hydrological
       regionalization of streams in Victoria, Australia, with implications
       for stream Ecology. Marine and Freshwater Research, 40(3), 303–326.

    """
    series = series.resample(freq).mean()
    cv = series.std() / series.mean()
    return cv


def cv_date_min(series):
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
    according to [richter_1996]_.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174.

    """
    data = series.groupby(series.index.year).idxmin().dropna().values
    data = DatetimeIndex(data).dayofyear.to_numpy(float)
    cv = data.std() / data.mean()
    return cv


def parde_seasonality(series, normalize=True):
    """Parde seasonality according to [parde_1933]_.

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
    Pardé seasonality is the difference between the maximum and minimum
    Pardé coefficient. A Pardé series consists of 12 Pardé coefficients,
    corresponding to 12 months. Pardé coefficient for, for example, January is
    its long‐term monthly mean groundwater head divided by the overall mean
    groundwater head.

    References
    ----------
    .. [parde_1933] Pardé, M. (1933). Fleuves et rivieres.

    """
    coefficients = parde_coefficients(series=series, normalize=normalize)
    return coefficients.max() - coefficients.min()


def parde_coefficients(series, normalize=True):
    """Parde coefficients for each month [parde_1933]_.

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
    Pardé seasonality is the difference between the maximum and minimum
    Pardé coefficient. A Pardé series consists of 12 Pardé coefficients,
    corresponding to 12 months. Pardé coefficient for, for example, January is
    its long‐term monthly mean groundwater head divided by the overall mean
    groundwater head.

    References
    ----------
    .. [parde_1933] Pardé, M. (1933). Fleuves et rivieres.

    """
    if normalize:
        series = _normalize(series)

    coefficients = series.groupby(series.index.month).mean() / series.mean()
    coefficients.index.name = "month"
    return coefficients


def _martens(series, normalize=True):
    """Functions for the Martens average seasonal fluctuation and
    interanual fluctuation.

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
    """

    if normalize:
        series = _normalize(series)

    s = series.resample("M")
    hl = s.min().groupby(s.min().index.year).nsmallest(3).groupby(
        level=0).mean()
    hw = s.max().groupby(s.max().index.year).nlargest(3).groupby(
        level=0).mean()

    return hl, hw


def avg_seasonal_fluctuation(series, normalize=False):
    """Classification according to [martens_2013]_.

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
    Mean annual difference between the averaged 3 highest monthly
    groundwater heads per year and the averaged 3 lowest monthly groundwater
    heads per year.

    Average seasonal fluctuation (s):

        s = MHW - MLW

    References
    ----------
    .. [martens_2013] Martens, K., van Camp, M., van Damme, D., & Walraevens,
       K. (2013). Groundwater dynamics converted to a groundwater
       classification as a tool for nature development programs in the
       dunes. Journal of Hydrology, 499, 236–246.

    """

    hl, hw = _martens(series, normalize=normalize)

    return hw.mean() - hl.mean()


def interannual_variation(series, normalize=False):
    """Interannual variation after [martens_2013]_.

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
    The average between the range in annually averaged 3 highest monthly
    groundwater heads and the range in annually averaged 3 lowest monthly
    groundwater heads.

    Inter-yearly variation of high and low water table (y):

        y = ((max_HW - min_HW) + (max_LW - min_LW)) / 2

    Warning: In this formulating the water table is references to a certain
    datum and positive, not as depth below the surface!

    References
    ----------
    .. [martens_2013] Martens, K., van Camp, M., van Damme, D., & Walraevens,
       K. (2013). Groundwater dynamics converted to a groundwater
       classification as a tool for nature development programs in the
       dunes. Journal of Hydrology, 499, 236–246.

    """

    hl, hw = _martens(series, normalize=normalize)

    return (hw.max() - hw.min()) + (hl.max() - hl.min()) / 2


def colwell_components(series, bins=11, freq="M", method="mean",
                       normalize=True):
    """Colwell predictability, constant, and contingency [colwell_1974]_.

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
    The difference between the sum of entropy for each time step and
    possible state of the seasonal cycle, and the overall entropy across all
    states and time steps, divided by the logarithm of the absolute number of
    possible states. Entropy according to definition in information theory,
    see reference for details.

    References
    ----------
    .. [colwell_1974] Colwell, R. K. (1974). Predictability Constancy and
       Contingency of periodic phenomena. Ecology, 55(5), 1148–1153.

    """
    # Prepare data and pivot table
    if normalize:
        series = _normalize(series)

    if method == "mean":
        series = series.resample(freq).mean().dropna()
    else:
        raise NotImplementedError

    series.name = "head"
    binned = cut(series, bins=bins, right=False, include_lowest=True,
                 labels=range(bins))
    df = pd.DataFrame(binned)
    df["time"] = df.index.month
    df["values"] = 1
    df = df.pivot_table(columns="head", index="time", aggfunc="sum",
                        values="values")

    # Count of rows and column items
    x = df.sum(axis=1)  # Time
    y = df.sum(axis=0)  # Head
    z = series.size  # Total number of observations

    hx = -(x / z * log(x / z)).sum()
    hy = - (y / z * log(y / z)).sum()
    hxy = - (df / z * log(df / z, where=df!=0)).sum().sum()

    # Compute final components
    p = 1 - (hxy - hy) / log(bins)  # Predictability
    c = 1 - hx / log(bins)  # Constancy
    m = (hx + hy - hxy) / log(bins)  # Contingency
    return p, c, m


def colwell_constancy(series, bins=11, freq="M", method="mean",
                      normalize=True):
    """Colwells constancy index after [colwell_1974]_.

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
    One minus the sum of entropy with respect to state, divided by the
    logarithm of the absolute number of possible states.

    References
    ----------
    .. [colwell_1974] Colwell, R. K. (1974). Predictability Constancy and
       Contingency of periodic phenomena. Ecology, 55(5), 1148–1153.

    """
    return \
        colwell_components(series=series, bins=bins, freq=freq, method=method,
                           normalize=normalize)[1]


def colwell_contingency(series, bins=11, freq="M", method="mean",
                        normalize=True):
    """Colwell contingency [colwell_1974]_

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
    The difference between the sum of entropy for each time step and
    possible state of the seasonal cycle, and the overall entropy across all
    states and time steps, divided by the logarithm of the absolute number of
    possible states. Entropy according to definition in information theory,
    see reference for details.

    References
    ----------
    .. [colwell_1974] Colwell, R. K. (1974). Predictability Constancy and
       Contingency of periodic phenomena. Ecology, 55(5), 1148–1153.

    """
    return \
        colwell_components(series=series, bins=bins, freq=freq, method=method,
                           normalize=normalize)[2]


def low_pulse_count(series, quantile=0.2):
    """Number of times the series drops below a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------
    int
        Number of times the series exceeds a certain threshold.

    Notes
    -----
    Number of times during which the groundwater head drops below a certain
    threshold. The threshold is defined as the 20th percentile of
    nonexceedance [richter_1996]_.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174.

    """
    h = series < series.quantile(quantile)
    return (h.astype(int).diff() > 0).sum()


def high_pulse_count(series, quantile=0.8):
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
    Number of times during which the groundwater head exceeds a certain
    threshold. The threshold is defined as the 80th percentile of
    nonexceedance.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174.

    """
    h = series > series.quantile(quantile)
    return (h.astype(int).diff() > 0).sum()


def low_pulse_duration(series, quantile=0.2):
    """Average duration of pulses where the head is below a certain threshold.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with DatetimeIndex and head values.
    quantile: float, optional
        Quantile used as a threshold.

    Returns
    -------

    Notes
    -----
    Average duration of pulses where the groundwater head drops below
    a certain threshold. The threshold is defined as the 20th percentile of
    non-exceedance.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    h = series < series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, NA).shift(-1).dropna().index
    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def high_pulse_duration(series, quantile=0.8):
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
    Average duration of pulses where the groundwater head drops exceeds a
    certain threshold. The threshold is defined as the 80th percentile of
    nonexceedance.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    h = series > series.quantile(quantile)
    sel = h.astype(int).diff().replace(0.0, NA).shift(-1).dropna().index
    return (diff(sel.to_numpy()) / Timedelta("1D"))[::2].mean()


def amplitude_range(series):
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


def rise_rate(series, normalize=False):
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

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    rises = difference[difference > 0]
    return rises.mean()


def fall_rate(series, normalize=False):
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
    Mean rate of negative changes in head from one day to the next,
    according to [richter_1996]_.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    falls = difference[difference < 0]
    return falls.mean()


def cv_rise_rate(series, normalize=False):
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
    Coefficient of Variation in riserate.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    rises = difference[difference > 0]
    return rises.std() / rises.mean()


def cv_fall_rate(series, normalize=False):
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
    Coefficient of Variation in fall rate.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174. https://doi.org/10.1046/j.1523‐1739.1996.10041163.x

    """
    if normalize:
        series = _normalize(series)

    difference = series.diff()
    falls = difference[difference < 0]
    return falls.std() / falls.mean()


def magnitude(series):
    """Difference of peak head to base head, divided by base head.

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

    (h_max - h_min ) / h_min

    References
    ----------
    .. [hannah_2000] Hannah, D. M., Smith, B. P. G., Gurnell, A. M.,
       & McGregor, G. R. (2000). An approach to hydrograph classification.
       Hydrological Processes, 14(2), 317–338.

    """

    return (series.max() - series.min()) / series.min()


def reversals_avg(series):
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
    Average annual number of rises and falls (i.e., change of sign) in daily
    head.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174.

    """
    reversals = (series.diff() > 0).astype(int).diff().replace(-1, 1)
    return reversals.resample("A").sum().mean()


def reversals_cv(series):
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
    Coefficient of Variation in annual number of rises and falls in daily head.

    References
    ----------
    .. [richter_1996] Richter, B. D. (1996). A method for assessing hydrologic
       alteration within ecosystems. Society for Conservation Biology, 10(4),
       1163–1174.

    """
    reversals = (series.diff() > 0).astype(int).diff().replace(-1, 1) \
        .resample("A").sum()
    return reversals.std() / reversals.mean()


def mean_annual_maximum(series, normalize=False):
    """Mean of annual maximum.

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

    References
    ----------
    .. [clausen_2000] Clausen, B., & Biggs, B. J. F. (2000). Flow variables
       for ecological studies in temperate streams—Groupings based on
       covariance. Journal of Hydrology, 237(3‐4), 184–197.

    """
    if normalize:
        series = _normalize(series)

    return series.resample("A").max().mean()


def bimodality_coefficient(series, normalize=True):
    """Bimodality coefficient after [Ellison_1987]_.

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
    Squared product moment skewness plus one, divided by product moment
    kurtosis.

        b = (skew **2 + 1 ) / kurtosis

    Adapted from the R "modes" package

    References
    ----------
    .. [Ellison_1987] Ellison, A. M. (1987). Effect of seed dimorphism on the
       density‐dependent dynamics of experimental populations of atriplex
       triangularis. American Journal of Botany, 74(8), 1280–1288.

    """
    if normalize:
        series = _normalize(series)
    series = series.dropna()
    n = series.size
    # Compute the skew for a finite sample
    skew = (1 / n) * sum((series - series.mean()) ** 3) / \
           (((1 / n) * sum((series - series.mean()) ** 2)) ** 1.5)
    skew *= (sqrt(n * (n - 1))) / (n - 2)

    # Compute the kurtosis for a finite sample
    kurt = (1 / n) * sum((series - series.mean()) ** 4) / \
           (((1 / n) * sum((series - series.mean()) ** 2)) ** 2) - 3
    kurt = ((n - 1) * ((n + 1) * kurt - 3 * (n - 1)) / ((n - 2) * (n - 3))) + 3

    return ((skew ** 2) + 1) / \
           (kurt + ((3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))))


def recession_constant(series, bins=20, normalize=False):
    """Recession constant after [kirchner_2009]_.

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
    Slope of the linear model fitted to percentile‐wise binned means in a
    log‐log plot of negative head versus negative head one time step ahead.

    References
    ----------
    .. [kirchner_2009] Kirchner, J. W. (2009). Catchments as simple dynamical
       systems: Catchment characterization, rainfall‐runoff modeling, and doing
       hydrology backward. Water Resources Research, 45, W02429.

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


def recovery_constant(series, bins=20, normalize=False):
    """Recovery constant after [kirchner_2009]_.

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
    Slope of the linear model fitted to percentile‐wise binned means in a
    log‐log plot of positive head versus positive head one time step ahead.

    References
    ----------
    .. [kirchner_2009] Kirchner, J. W. (2009). Catchments as simple dynamical
       systems: Catchment characterization, rainfall‐runoff modeling, and doing
       hydrology backward. Water Resources Research, 45, W02429.


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


def duration_curve_slope(series, l=0.1, u=0.9, normalize=True):
    """Slope of the duration curve between percentile l and u.

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
    Slope of the duration curve (analogue flow duration curve for streamflow)
    between percentile l and u.

    References
    ----------
    .. [oudin_2010] Oudin, L., Kay, A., Andréassian, V., & Perrin, C. (2010).
       Are seemingly physically similar catchments truly hydrologically
       similar? Water Resources Research, 46, W11558.

    """
    if normalize:
        series = _normalize(series)

    s = series[(series.quantile(l) > series) & (series < series.quantile(
        u))].sort_values()
    s.index = arange(s.size) / s.size
    return linregress(s.index, s.values).slope


def duration_curve_range(series, l=0.1, u=0.9, normalize=True):
    """Range of the duration curve between the percentile l and u.

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

    References
    ----------
    .. [richards_1990] Richards, R. P. (1990). Measures of Flow Variability
       and a New Flow‐Based Classification of Great Lakes Tributaries.
       Journal of Great Lakes Research, 16(1), 53–70.

    """
    if normalize:
        series = _normalize(series)

    return series.quantile(u) - series.quantile(l)


def richards_pathlength(series, normalize=True):
    """The path length of the time series, standardized by time series length.

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
    The path length of the time series, standardized by time series length
    and the median head.

    References
    ----------
    .. [baker_2004] Baker, D. B., Richards, P., Loftus, T. T., & Kramer,
       J. W. (2004). A new flashiness index: Characteristics and applications
       to midwestern rivers and streams. Journal of the American Water
       Resources Association, 40(2), 503–522.

    """
    if normalize:
        series = _normalize(series)

    series = series.dropna()
    dt = diff(series.index.to_numpy()) / Timedelta("1D")
    dh = series.diff().dropna()
    # sum(dt) is more fair with irregular time series
    return sum(sqrt(dh ** 2 + dt ** 2)) / sum(dt)


def richards_baker_index(series, normalize=True):
    """Richards-Baker index according to [baker_2004]_.

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
    Sum of absolute values of day‐to‐day changes in head divided by the sum
    of scaled daily head. Equivalent the Richards Pathlength without the
    time component.

    References
    ----------
    .. [baker_2004] Baker, D. B., Richards, P., Loftus, T. T., & Kramer,
       J. W. (2004). A new flashiness index: Characteristics and applications
       to midwestern rivers and streams. Journal of the American Water
       Resources Association, 40(2), 503–522.

    """
    if normalize:
        series = _normalize(series)

    return series.diff().dropna().abs().sum() / series.sum()


def _baseflow(series, normalize=True):
    """Baseflow function for the baseflow index and stability

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
    ht[ht > series.resample("D").mean().loc[ht.index]] = \
        series.resample("D").mean()

    return series, ht


def baseflow_index(series, normalize=True):
    """Baseflow index according to [wmo_2008]_.

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
    Adapted analogously to its application in streamflow. Here, a baseflow
    time series is separated from a 5‐day minimum groundwater head in a
    moving window. BFI equals the total sum of heads of original time series
    divided by the total sum of heads from the baseflow type of time series.

    References
    ----------
    .. [wmo_2008] WMO (2008). Manual on Low‐Flow Estimation and Prediction.
       Geneva, Switzerland: World Meteorological Organization.

    """

    series, ht = _baseflow(series, normalize=normalize)

    return series.resample("D").mean().sum() / ht.sum()


def baseflow_stability(series, normalize=True):
    """Baseflow stability after [heudorfer_2019]_.

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
    Originally developed for streamflow, here the Base Flow Index
    algorithm is analogously adapted to groundwater time series as a filter
    to separate the slow component (“baseflow”) of the time series. Then,
    the mean annual baseflow is calculated. Base Flow Stability is the
    difference of maximum and minimum annual baseflow.

    References
    ----------
    .. [heudorfer_2019] Heudorfer, B., Haaf, E., Stahl, K., & Barthel, R.
       (2019). Index‐based characterization and quantification of groundwater
       dynamics. Water Resources Research, 55, 5575–5592.

    """

    series, ht = _baseflow(series, normalize=normalize)

    return ht.resample("A").mean().max() - ht.resample("A").mean().min()


def hurst_exponent(series):
    """Hurst exponent according to [wang_2006]_.

    Returns
    -------

    Notes
    -----
    The slope of a linear model fitted to the relationship between the
    sample size and the logarithmized sample range of k contiguous
    subsamples from the time series.

    References
    ----------
    .. [wang_2006] Wang, X., Smith, K., & Hyndman, R. (2006).
       Characteristic‐based clustering for time series data. Data Mining and
       Knowledge Discovery, 13(3), 335–364.

    """
    return NotImplementedError


def autocorr(series, freq="w"):
    """Lag where the first peak in the autocorrelation function occurs.

    Returns
    -------

    Notes
    -----
    Lag where the first peak in the autocorrelation function occurs.

    References
    ----------
    .. [wang_2006] Wang, X., Smith, K., & Hyndman, R. (2006).
       Characteristic‐based clustering for time series data. Data Mining and
       Knowledge Discovery, 13(3), 335–364.

    """
    return NotImplementedError


def lyapunov_exponent(series):
    """The exponential rate of divergence of nearby data points
    [hilborn_1994]_.

    Returns
    -------

    Notes
    -----
    The exponential rate of divergence of nearby data points when moving
    away in time from a certain data point in the series. Iteratively
    estimated for every point in the time series, then averaged.

    References
    ----------
    .. [hilborn_1994] Hilborn, R. C. (1994). Chaos and Nonlinear Dynamics: An
       Introduction for Scientists and Engineers. New York: Oxford University
       Press.

    """
    return NotImplementedError


def peak_timescale(series):
    """Area under peak divided by difference of peak head to peak base.

    Returns
    -------

    Notes
    -----
    Area under peak divided by difference of peak head to peak base,
    averaged over all peaks.

    References
    ----------
    .. [gaal_2012] Gaál, L., Szolgay, J., Kohnová, S., Parajka, J., Merz, R.,
       Viglione, A., & Blöschl, G. (2012). Flood timescales: Understanding
       the interplay of climate and catchment processes through comparative
       hydrology. Water Resources Research, 48, W04511.

    """
    return NotImplementedError


def excess_mass(series):
    """Test statistic of the dip test, after [hartigan_1985]_.

    Returns
    -------

    Notes
    -----
    Test statistic of the dip test; maximum distance between the empirical
    distribution and the best fitting unimodal distribution. By default the
    best fitting distribution is the uniform.

    References
    ----------
    .. [hartigan_1985] Hartigan, J. A., & Hartigan, P. M. (1985). The dip
       test of unimodality. The Annals of Statistics, 13(1), 70–84.

    """
    return NotImplementedError


def critical_bandwidth(series):
    """Test statistic of the Silverman test, after [silverman_1981]_.

    Returns
    -------

    Notes
    -----
    Test statistic of the Silverman test; minimum kernel bandwidth required
    to create a unimodal distribution estimated by fitting a Kernel Density
    Estimation.

    References
    ----------
    .. [silverman_1981] Silverman, B. W. (1981). Using kernel density
       estimates to investigate multimodality. Journal of The Royal
       Statistical Society Series B‐ Statistical Methodology, 43(1), 97–99.

    """
    return NotImplementedError


def peak_base_time(series):
    """Difference between peak and base head, standardized by duration of peak.

    Returns
    -------

    Notes
    -----
    Difference between peak and base head, standardized by duration of peak.

    References
    ----------
    .. [heudorfer_2019] Heudorfer, B., Haaf, E., Stahl, K., & Barthel, R.
       (2019). Index‐based characterization and quantification of groundwater
       dynamics. Water Resources Research, 55, 5575–5592.
       https://doi.org/10.1029/2018WR024418

    """
    return NotImplementedError


def summary(series, signatures=None):
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
    idx = pd.date_range("2000", "2010")
    head = pd.Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    ps.stats.signatures.summary(head)

    """
    if signatures is None:
        signatures = __all__

    data = pd.Series(index=signatures, dtype=float)

    for signature in signatures:
        func = getattr(ps.stats.signatures, signature)
        data.loc[signature] = func(series)

    return data
