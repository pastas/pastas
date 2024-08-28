"""The following methods may be used to calculate the crosscorrelation and
autocorrelation for a time series.

These methods are 'special' in the sense that they are able to deal with irregular
time steps often observed in hydrological time series.
"""

# Type Hinting
from typing import Tuple, Union

from numpy import (
    append,
    arange,
    array,
    average,
    corrcoef,
    diff,
    empty_like,
    exp,
    inf,
    nan,
    ones,
    pi,
    sqrt,
)
from pandas import DataFrame, Index, Series, Timedelta, to_timedelta
from scipy.stats import norm

from pastas.typing import ArrayLike

from ..decorators import njit


def acf(
    x: Series,
    lags: ArrayLike = 365,
    bin_method: str = "rectangle",
    bin_width: float = 0.5,
    max_gap: float = inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
) -> Union[Series, DataFrame]:
    """Calculate the autocorrelation function for irregular time steps.

    Parameters
    ----------
    x: pandas.Series
        Pandas Series containing the values to calculate the cross-correlation on.
        The index has to be a Pandas.DatetimeIndex.
    lags: array_like, optional
        numpy array containing the lags in days for which the cross-correlation if
        calculated. Defaults is all lags from 1 to 365 days.
    bin_method: str, optional
        method to determine the type of bin. Options are "rectangle" (default),
        "gaussian" and "regular" (for regular timesteps).
    bin_width: float, optional
        number of days used as the width for the bin to calculate the correlation.
    max_gap: float, optional
        Maximum time step gap in the data. All time steps above this gap value are
        not used for calculating the average time step. This can be helpful when
        there is a large gap in the data that influences the average time step.
    min_obs: int, optional
        Minimum number of observations in a bin to determine the correlation.
    full_output: bool, optional
        If True, also estimated uncertainties are returned. Default is False.
    alpha: float, optional
        alpha level to compute the confidence interval (e.g., 1-alpha).

    Returns
    -------
    result: pandas.Series or pandas.DataFrame
        If full_output=True, a DataFrame with columns "acf", "conf", and "n",
        containning the autocorrelation function, confidence intervals (depends on
        alpha), and the number of samples n used to compute these, respectively. If
        full_output=False, only the ACF is returned.

    Notes
    -----
    Calculate the autocorrelation function for irregular timesteps based on the
    slotting technique. Different methods (kernels) to bin the data are available.
    Method here is based on :cite:t:`rehfeld_comparison_2011`.

    If the time series have regular time step we recommend to use the acf method from
    the Statsmodels package.

    Examples
    --------
    For example, to estimate the autocorrelation for every second lag up to lags of
    one year:

    >>> acf = ps.stats.acf(x, lags=np.arange(1.0, 366.0, 2.0))

    See Also
    --------
    pastas.stats.ccf
    statsmodels.api.tsa.acf
    """
    c = ccf(
        x=x,
        y=x,
        lags=lags,
        bin_method=bin_method,
        bin_width=bin_width,
        max_gap=max_gap,
        min_obs=min_obs,
        full_output=full_output,
        alpha=alpha,
    )
    c.name = "ACF"
    if full_output:
        return c.rename(columns={"ccf": "acf"})
    else:
        return c


def ccf(
    x: Series,
    y: Series,
    lags: ArrayLike = 365,
    bin_method: str = "rectangle",
    bin_width: float = 0.5,
    max_gap: float = inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
) -> Union[Series, DataFrame]:
    """Method to compute the cross-correlation for irregular time series.

    Parameters
    ----------
    x,y: pandas.Series
        Pandas Series containing the values to calculate the cross-correlation on.
        The index has to be a Pandas.DatetimeIndex.
    lags: array_like, optional
        numpy array containing the lags in days for which the cross-correlation is
        calculated. Defaults is all lags from 1 to 365 days.
    bin_method: str, optional
        method to determine the type of bin. Options are "rectangle" (default),
        "gaussian" and "regular" (for regular timesteps).
    bin_width: float, optional
        number of days used as the width for the bin to calculate the correlation.
    max_gap: float, optional
        Maximum timestep gap in the data. All timesteps above this gap value are not
        used for calculating the average timestep. This can be helpful when there is
        a large gap in the data that influences the average timestep.
    min_obs: int, optional
        Minimum number of observations in a bin to determine the correlation.
    full_output: bool, optional
        If True, also estimated uncertainties are returned. Default is False.
    alpha: float
        alpha level to compute the confidence interval (e.g., 1-alpha).

    Returns
    -------
    result: pandas.Series or pandas.DataFrame
        If full_output=True, a DataFrame with columns "ccf", "conf", and "n",
        containning the cross-correlation function, confidence intervals (depends on
        alpha), and the number of samples n used to compute these, respectively. If
        full_output=False, only the CCF is returned.

    Examples
    --------
    >>> ccf = ps.stats.ccf(x, y, bin_method="gaussian")
    """
    # prepare the time indices for x and y
    if x.index.inferred_freq and y.index.inferred_freq:
        bin_method = "regular"
    elif bin_method == "regular":
        raise Warning(
            "time series does not have regular time steps, choose different bin_method."
        )

    x, t_x, dt_x_mu = _preprocess(x, max_gap=max_gap)
    y, t_y, dt_y_mu = _preprocess(y, max_gap=max_gap)
    dt_mu = max(dt_x_mu, dt_y_mu)  # The mean time step from both series

    if isinstance(lags, int) and bin_method == "regular":
        lags = arange(int(dt_mu), lags + 1, int(dt_mu), dtype=float)
    elif isinstance(lags, int):
        lags = arange(1.0, lags + 1, dtype=float)
    elif isinstance(lags, list):
        lags = array(lags, dtype=float)

    if bin_method == "rectangle":
        c, b = _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "gaussian":
        c, b = _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "regular":
        c, b = _compute_ccf_regular(arange(1.0, len(lags) + 1), x, y)
    else:
        raise NotImplementedError

    conf = norm.ppf(1 - alpha / 2.0) / sqrt(b)
    result = DataFrame(
        data={"ccf": c, "conf": conf, "n": b},
        dtype=float,
        index=Index(to_timedelta(lags, unit="D"), name="Lags"),
    )

    result = result.where(result.n > min_obs).dropna()

    if full_output:
        return result
    else:
        return result.ccf


def _preprocess(x: Series, max_gap: float) -> Tuple[ArrayLike, ArrayLike, float]:
    """Internal method to preprocess the time series."""
    dt = x.index.to_series().diff().dropna().values / Timedelta(1, "D")
    dt_mu = dt[dt < max_gap].mean()  # Deal with big gaps if present
    if int(dt_mu) == 0:
        dt_mu = 1  # Prevent division by zero error
    t = dt.cumsum()

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()

    return x, t, dt_mu


@njit
def _compute_ccf_rectangle(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> Tuple[ArrayLike, ArrayLike]:
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    n = len(t_x)

    for k in range(len(lags)):
        cl = 0.0
        b_sum = 0.0
        for i in range(n):
            for j in range(n):
                d = abs(t_x[i] - t_y[j]) - lags[k]
                if abs(d) <= bin_width:
                    cl += x[i] * y[j]
                    b_sum += 1.0
        if b_sum == 0.0:
            c[k] = nan
            b[k] = 1e-16  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum / 2.0  # divide by 2 because we over count in for-loop
    return c, b


@njit
def _compute_ccf_gaussian(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> Tuple[ArrayLike, ArrayLike]:
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    n = len(t_x)

    den1 = -2 * bin_width**2  # denominator 1
    den2 = sqrt(2 * pi * bin_width)  # denominator 2

    for k in range(len(lags)):
        cl = 0.0
        b_sum = 0.0

        for i in range(n):
            for j in range(n):
                d = t_x[i] - t_y[j] - lags[k]
                d = exp(d**2 / den1) / den2
                cl += x[i] * y[j] * d
                b_sum += d
        if b_sum == 0.0:
            c[k] = nan
            b[k] = 1e-16  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum / 2.0  # divide by 2 because we over count in for-loop
    return c, b


def _compute_ccf_regular(
    lags: ArrayLike, x: ArrayLike, y: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    c = empty_like(lags)
    for i, lag in enumerate(lags):
        c[i] = corrcoef(x[: -int(lag)], y[int(lag) :])[0, 1]
    b = len(x) - lags
    return c, b


def mean(x: Series, weighted: bool = True, max_gap: int = 30) -> ArrayLike:
    """Method to compute the (weighted) mean of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the mean weight. Default
        value is 90 days.

    Notes
    -----
    The (weighted) mean for a time series x is computed as:

    .. math:: \\bar{x} = \\sum_{i=1}^{N} w_i x_i

    where :math:`w_i` are the weights, taken as the time step between observations,
    normalized by the sum of all time steps.
    """
    w = _get_weights(x, weighted=weighted, max_gap=max_gap)
    return average(x.to_numpy(), weights=w)


def var(x: Series, weighted: bool = True, max_gap: int = 30) -> ArrayLike:
    """Method to compute the (weighted) variance of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the mean weight. Default
        value is 90 days.

    Notes
    -----
    The (weighted) variance for a time series x is computed as:

    .. math:: \\sigma_x^2 = \\sum_{i=1}^{N} w_i (x_i - \\bar{x})^2

    where :math:`w_i` are the weights, taken as the time step between observations,
    normalized by the sum of all time steps. Note how weighted mean (:math:`\\bar{
    x}`) is used in this formula.
    """
    w = _get_weights(x, weighted=weighted, max_gap=max_gap)
    mu = average(x.to_numpy(), weights=w)
    sigma = (x.size / (x.size - 1) * w * (x.to_numpy() - mu) ** 2).sum()
    return sigma


def std(x: Series, weighted: bool = True, max_gap: int = 30) -> ArrayLike:
    """Method to compute the (weighted) variance of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the mean weight. Default
        value is 90 days.

    See Also
    --------
    ps.stats.mean, ps.stats.var
    """
    return sqrt(var(x, weighted=weighted, max_gap=max_gap))


# Helper functions


def _get_weights(x: Series, weighted: bool = True, max_gap: int = 30) -> ArrayLike:
    """Helper method to compute the weights as the time step between obs.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the mean weight. Default
        value is 30 days.
    """
    if weighted:
        w = append(0.0, diff(x.index.to_numpy()) / Timedelta("1D"))
        w[w > max_gap] = max_gap
    else:
        w = ones(x.index.size)
    w /= w.sum()
    return w
