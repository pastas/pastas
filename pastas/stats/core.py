"""The following methods may be used to calculate the crosscorrelation and
autocorrelation for a time series.

These methods are 'special' in the sense that they are able to deal with
irregular time steps often observed in hydrological time series.
"""

from numpy import (append, arange, array, average, corrcoef, diff, empty_like,
                   exp, inf, nan, ones, pi, sqrt)
from pandas import DataFrame, Timedelta, TimedeltaIndex
from scipy.stats import norm

from ..decorators import njit
from ..utils import check_numba


def acf(x, lags=365, bin_method='rectangle', bin_width=0.5, max_gap=inf,
        min_obs=20, full_output=False, alpha=0.05):
    """Calculate the autocorrelation function for irregular time steps.

    Parameters
    ----------
    x: pandas.Series
        Pandas Series containing the values to calculate the
        cross-correlation for. The index has to be a Pandas.DatetimeIndex
    lags: array_like, optional
        numpy array containing the lags in days for which the
        cross-correlation if calculated. [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12,
        13, 14, 30, 61, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    bin_method: str, optional
        method to determine the type of bin. Options are "rectangle" (default),
        and  "gaussian".
    bin_width: float, optional
        number of days used as the width for the bin to calculate the
        correlation. By default these values are chosen based on the
        bin_method and the average time step (dt_mu). That is 0.5dt_mu when
        bin_method="rectangle" and 0.25dt_mu when bin_method="gaussian".
    max_gap: float, optional
        Maximum time step gap in the data. All time steps above this gap value
        are not used for calculating the average time step. This can be
        helpful when there is a large gap in the data that influences the
        average time step.
    min_obs: int, optional
        Minimum number of observations in a bin to determine the correlation.
    full_output: bool, optional
        If True, also estimated uncertainties are returned. Default is False.
    alpha: float
        alpha level to compute the confidence interval (e.g., 1-alpha).

    Returns
    -------
    c: pandas.Series or pandas.DataFrame
        The autocorrelation function for the provided lags.

    Notes
    -----
    Calculate the autocorrelation function for irregular timesteps based on
    the slotting technique. Different methods (kernels) to bin the data are
    available.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    Tip
    ---
    If the time series have regular time step we recommend to use the acf
    method from the Statsmodels package.

    Examples
    --------
    For example, to estimate the autocorrelation for every second lag up to
    lags of one year:

    >>> acf = ps.stats.acf(x, lags=np.arange(1.0, 366.0, 2.0))

    See Also
    --------
    pastas.stats.ccf
    statsmodels.api.tsa.acf
    """
    c = ccf(x=x, y=x, lags=lags, bin_method=bin_method, bin_width=bin_width,
            max_gap=max_gap, min_obs=min_obs, full_output=full_output,
            alpha=alpha)
    c.name = "ACF"
    if full_output:
        return c.rename(columns={"ccf": "acf"})
    else:
        return c


def ccf(x, y, lags=365, bin_method='rectangle', bin_width=0.5,
        max_gap=inf, min_obs=20, full_output=False, alpha=0.05):
    """Method to compute the cross-correlation for irregular time series.

    Parameters
    ----------
    x,y: pandas.Series
        Pandas Series containing the values to calculate the
        cross-correlation for. The index has to be a Pandas.DatetimeIndex
    lags: array_like, optional
        numpy array containing the lags in days for which the
        cross-correlation is calculated. Default [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 30, 61, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    bin_method: str, optional
        method to determine the type of bin. Options are "rectangle" (default),
        "gaussian" and "regular" (for regular timesteps).
    bin_width: float, optional
        number of days used as the width for the bin to calculate the
        correlation. By default these values are chosen based on the
        bin_method and the average time step (dt_mu). That is 0.5dt_mu when
        bin_method="rectangle" and 0.25dt_mu when bin_method="gaussian".
    max_gap: float, optional
        Maximum timestep gap in the data. All timesteps above this gap value
        are not used for calculating the average timestep. This can be
        helpful when there is a large gap in the data that influences the
        average timestep.
    min_obs: int, optional
        Minimum number of observations in a bin to determine the correlation.
    full_output: bool, optional
        If True, also estimated uncertainties are returned. Default is False.
    alpha: float
        alpha level to compute the confidence interval (e.g., 1-alpha).

    Returns
    -------
    c: pandas.Series or pandas.DataFrame
        The Cross-correlation function.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    Tip
    ---
    This method will be significantly faster when Numba is installed. Check
    out the [Numba project here](https://numba.pydata.org)

    Examples
    --------
    >>> ccf = ps.stats.ccf(x, y, bin_method="gaussian")
    """
    # prepare the time indices for x and y
    if x.index.inferred_freq and y.index.inferred_freq:
        bin_method = "regular"
    elif bin_method == "regular":
        raise Warning("time series does not have regular time steps, "
                      "choose different bin_method")

    x, t_x, dt_x_mu = _preprocess(x, max_gap=max_gap)
    y, t_y, dt_y_mu = _preprocess(y, max_gap=max_gap)
    dt_mu = max(dt_x_mu, dt_y_mu)  # Mean time step from both series

    if isinstance(lags, int) and bin_method == "regular":
        lags = arange(int(dt_mu), lags + 1, int(dt_mu), dtype=float)
    elif isinstance(lags, int):
        lags = arange(1.0, lags + 1, dtype=float)
    elif isinstance(lags, list):
        lags = array(lags, dtype=float)

    if bin_method == "rectangle":
        if bin_width is None:
            bin_width = 0.5 * dt_mu
        check_numba()
        c, b = _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "gaussian":
        if bin_width is None:
            bin_width = 0.25 * dt_mu
        check_numba()
        c, b = _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "regular":
        c, b = _compute_ccf_regular(arange(1.0, len(lags) + 1), x, y)
    else:
        raise NotImplementedError

    std = norm.ppf(1 - alpha / 2.) / sqrt(b)
    result = DataFrame(data={"ccf": c, "stderr": std, "n": b},
                       index=TimedeltaIndex(lags, unit="D", name="Lags"))

    result = result.where(result.n > min_obs).dropna()

    if full_output:
        return result
    else:
        return result.ccf


def _preprocess(x, max_gap):
    """Internal method to preprocess the time series."""
    dt = x.index.to_series().diff().dropna().values / Timedelta(1, "D")
    dt_mu = dt[dt < max_gap].mean()  # Deal with big gaps if present
    t = dt.cumsum()

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()

    return x, t, dt_mu


@njit
def _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width=0.5):
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    l = len(lags)
    n = len(t_x)

    for k in range(l):
        cl = 0.
        b_sum = 0.
        for i in range(n):
            for j in range(n):
                d = abs(t_x[i] - t_y[j]) - lags[k]
                if abs(d) <= bin_width:
                    cl += x[i] * y[j]
                    b_sum += 1
        if b_sum == 0.:
            c[k] = nan
            b[k] = 0.01  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum / 2  # divide by 2 because we over count in for-loop
    return c, b


@njit
def _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width=0.5):
    """Internal numba-optimized method to compute the ccf."""
    c = empty_like(lags)
    b = empty_like(lags)
    l = len(lags)
    n = len(t_x)

    den1 = -2 * bin_width ** 2  # denominator 1
    den2 = sqrt(2 * pi * bin_width)  # denominator 2

    for k in range(l):
        cl = 0.
        b_sum = 0.

        for i in range(n):
            for j in range(n):
                d = t_x[i] - t_y[j] - lags[k]
                d = exp(d ** 2 / den1) / den2
                cl += x[i] * y[j] * d
                b_sum += d
        if b_sum == 0.:
            c[k] = nan
            b[k] = 0.01  # Prevent division by zero error
        else:
            c[k] = cl / b_sum
            b[k] = b_sum / 2  # divide by 2 because we over count in for-loop
    return c, b


def _compute_ccf_regular(lags, x, y):
    c = empty_like(lags)
    for i, lag in enumerate(lags):
        c[i] = corrcoef(x[:-int(lag)], y[int(lag):])[0, 1]
    b = len(x) - lags
    return c, b


def mean(x, weighted=True, max_gap=30):
    """Method to compute the (weighted) mean of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    The (weighted) mean for a time series x is computed as:

    .. math:: \\bar{x} = \\sum_{i=1}^{N} w_i x_i

    where :math:`w_i` are the weights, taken as the time step between
    observations, normalized by the sum of all time steps.
    """
    w = _get_weights(x, weighted=weighted, max_gap=max_gap)
    return average(x.to_numpy(), weights=w)


def var(x, weighted=True, max_gap=30):
    """Method to compute the (weighted) variance of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    The (weighted) variance for a time series x is computed as:

    .. math:: \\sigma_x^2 = \\sum_{i=1}^{N} w_i (x_i - \\bar{x})^2

    where :math:`w_i` are the weights, taken as the time step between
    observations, normalized by the sum of all time steps. Note how
    weighted mean (:math:`\\bar{x}`) is used in this formula.
    """
    w = _get_weights(x, weighted=weighted, max_gap=max_gap)
    mu = average(x.to_numpy(), weights=w)
    sigma = (x.size / (x.size - 1) * w * (x.to_numpy() - mu) ** 2).sum()
    return sigma


def std(x, weighted=True, max_gap=30):
    """Method to compute the (weighted) variance of a time series.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    See Also
    --------
    ps.stats.mean, ps.stats.var
    """
    return sqrt(var(x, weighted=weighted, max_gap=max_gap))


# Helper functions

def _get_weights(x, weighted, max_gap=30):
    """Helper method to compute the weights as the time step between obs.

    Parameters
    ----------
    x: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 30 days.
    """
    if weighted:
        w = append(0.0, diff(x.index.to_numpy()) / Timedelta("1D"))
        w[w > max_gap] = max_gap
    else:
        w = ones(x.index.size)
    w /= w.sum()
    return w
