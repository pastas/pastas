"""The following methods may be used to calculate the crosscorrelation and
autocorrelation for a time series.

These methods are 'special' in the sense that they are able to deal with irregular
time steps often observed in hydrological time series.
"""

from logging import getLogger

from numba import prange
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
    ndarray,
    ones,
    pi,
    sqrt,
)
from pandas import DataFrame, Index, Series, Timedelta, to_timedelta
from scipy.stats import norm

from pastas.typing import ArrayLike

from ..decorators import njit

logger = getLogger(__name__)


def acf(
    x: Series,
    lags: ArrayLike = 365,
    bin_method: str = "regular",
    bin_width: float = 0.5,
    max_gap: float = inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
    fallback_bin_method: str = "gaussian",
) -> Series | DataFrame:
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
        method to determine the type of bin. Options are "regular" for regular data
        (default), and "gaussian" and "rectangle" for irregular data.
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
        containing the autocorrelation function, confidence intervals (depends on
        alpha), and the number of samples n used to compute these, respectively. If
        full_output=False, only the ACF is returned.

    Notes
    -----
    The ACF method primarily tries to estimate the autocorrelation using common
    techniques if the time step between the measurements is regular. If the time step
    is irregular, the method falls back to an alternative method to calculate the
    autocorrelation function for irregular timesteps based on the slotting technique
    :cite:t:`rehfeld_comparison_2011`. Different methods (kernels) to bin the data are
    available.

    Estimating the autocorrelation for irregular time steps can be challenging.
    Depending on the data and the binning method and settings used, the correlation can
    be above 1 or below -1. If this occurs, a warning is raised.

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
        fallback_bin_method=fallback_bin_method,
    )
    # drop value for lag=0 by default, unless explicitly included
    if c.index[0] == Timedelta(0) and isinstance(lags, int):
        c = c.drop(c.index[0])
    c.name = "ACF"
    if full_output:
        return c.rename(columns={"ccf": "acf"})
    else:
        return c


def ccf(
    x: Series,
    y: Series,
    lags: ArrayLike = 365,
    bin_method: str = "regular",
    bin_width: float = 0.5,
    max_gap: float = inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
    fallback_bin_method: str = "gaussian",
) -> Series | DataFrame:
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
        method to determine the type of bin. Options are "regular" for regular data
        (default), and "gaussian" and "rectangle" for irregular data.
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
    fallback_bin_method: str, optional
        method to determine the type of bin used to compute the correlations if the
        data has irregular time steps between the measurements. Options are "gaussian"
        (default) and "rectangle" .

    Returns
    -------
    result: pandas.Series or pandas.DataFrame
        If full_output=True, a DataFrame with columns "ccf", "conf", and "n",
        containing the cross-correlation function, confidence intervals (depends on
        alpha), and the number of samples n used to compute these, respectively. If
        full_output=False, only the CCF is returned.

    Examples
    --------
    >>> ccf = ps.stats.ccf(x, y, bin_method="gaussian")

    Notes
    -----
    The CCF method primarily tries to estimate the correlation using common
    techniques if the time step between the measurements is regular. If the time step
    is irregular, the method falls back to an alternative method to calculate the
    correlation function for irregular timesteps based on the slotting technique
    :cite:t:`rehfeld_comparison_2011`. Different methods (kernels) to bin the data are
    available.

    Estimating the correlation for irregular time steps can be challenging. Depending
    on the data and the binning method and settings used, the correlation can be above
    1 or below -1. If this occurs, a warning is raised.

    """
    # Check if the time series have regular time steps
    if (
        not x.index.inferred_freq
        and not y.index.inferred_freq
        and bin_method == "regular"
    ):
        msg = (
            f"time series does not have regular time steps, the fallback_bin_method"
            f"'{fallback_bin_method}' is applied"
        )
        logger.warning(msg)
        bin_method = fallback_bin_method

    # prepare the time indices for x and y
    x, t_x, dt_x_mu = _preprocess(x, max_gap=max_gap)
    y, t_y, dt_y_mu = _preprocess(y, max_gap=max_gap)
    dt_mu = max(dt_x_mu, dt_y_mu)  # The mean time step from both series

    if isinstance(lags, int) and bin_method == "regular":
        lags = arange(0, lags + 1, int(dt_mu), dtype=float)
    elif isinstance(lags, int):
        lags = arange(0, lags + 1, dtype=float)
    elif isinstance(lags, list):
        lags = array(lags, dtype=float)
    elif isinstance(lags, ndarray):
        # ensure dtype float otherwise numba will
        # create integer arrays for the results
        lags = lags.astype(float)

    if bin_method == "rectangle":
        c, b = _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "gaussian":
        c, b = _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "regular":
        c, b = _compute_ccf_regular(lags, x, y)
    else:
        raise NotImplementedError

    conf = norm.ppf(1 - alpha / 2.0) / sqrt(b)
    result = DataFrame(
        data={"ccf": c, "conf": conf, "n": b},
        dtype=float,
        index=Index(to_timedelta(lags, unit="D"), name="Lags"),
    )

    result = result.where(result.n > min_obs).dropna()

    # Raise a warning if the correlation is above 1 or below -1
    # NOTE: Using 1.01 to avoid excessive warnings when using binning methods for
    # irregular timesteps. In those cases correlations sometimes exceed 1 by a small
    # amount.
    if (result.ccf.abs() > 1.01).any():
        msg = (
            "The correlation is above 1 or below -1. This can occur due to the "
            "binning method used. Please check the data and the binning method and "
            "use the autocorrelation function with extreme care."
        )
        logger.warning(msg)

    if full_output:
        return result
    else:
        return result.ccf


def _preprocess(x: Series, max_gap: float) -> tuple[ArrayLike, ArrayLike, float]:
    """Internal method to preprocess the time series."""
    dt = x.index.to_series().diff().dropna().values / Timedelta(1, "D")
    dt_mu = dt[dt < max_gap].mean()  # Deal with big gaps if present
    dt_mu = max(dt_mu, 1)  # Prevent division by zero error
    t = dt.cumsum()

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()

    return x, t, dt_mu


@njit(parallel=True, nogil=True, cache=True)
def _compute_ccf_rectangle(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    n = len(t_x)

    for k in prange(len(lags)):
        cl = 0.0
        b_sum = 0.0
        lag_k = lags[k]
        # traverse the lower diagonal of NxN matrix: np.dot(x.T, y)
        for j in range(n):
            yj = y[j]
            t_yj = t_y[j]
            for i in range(j, n):
                d = abs(t_x[i] - t_yj) - lag_k
                if abs(d) <= bin_width:
                    cl += x[i] * yj
                    b_sum += 1.0
        if b_sum == 0.0:
            c[k] = nan
            b[k] = 1e-16  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum
    return c, b


@njit(parallel=True, nogil=True, cache=True)
def _compute_ccf_gaussian(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    n = len(t_x)

    den1 = -2 * bin_width**2  # denominator 1
    den2 = sqrt(2 * pi * bin_width)  # denominator 2
    six_den2 = 6 * den2  # six std. dev.
    for k in prange(len(lags)):
        cl = 0.0
        b_sum = 0.0
        lag_k = lags[k]
        # traverse the lower diagonal of NxN matrix: np.dot(x.T, y)
        for j in range(n):
            t_yj = t_y[j]
            yj = y[j]
            for i in range(j, n):
                dtlag = t_x[i] - t_yj - lag_k
                if abs(dtlag) < six_den2:
                    d = exp(dtlag**2 / den1) / den2
                    # if d > 1e-5:
                    cl += x[i] * yj * d
                    b_sum += d
        if b_sum == 0.0:
            c[k] = nan
            b[k] = 1e-16  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum
    return c, b


def _compute_ccf_regular(
    lags: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    c = empty_like(lags)
    n = len(x)
    for i in range(len(lags)):
        lag = int(lags[i])
        if lag < n:
            # flip x, y to match numpy/scipy correlate output order
            c[i] = corrcoef(y[: n - lag], x[lag:])[0, 1]
        else:
            c[i] = nan
    b = n - lags
    b[b <= 0] = 1e-16  # Prevent division by zero error
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
