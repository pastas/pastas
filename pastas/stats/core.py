"""Module containing methods to calculate statistics.

Compute core statistics such as variance, mean, standard deviation, moment, cross-
and autocorrelation for time series. These methods are 'special' in the sense that
they are able to deal with irregular time steps often observed in hydrological time series.
"""

from logging import getLogger

import numpy as np
from pandas import DataFrame, DatetimeIndex, Index, Series, Timedelta, to_timedelta
from scipy.stats import norm

from .._options import options
from ..decorators import deprecate_args_or_kwargs, njit
from ..typing import ArrayLike

try:
    from numba import prange
except ImportError:
    prange = range

logger = getLogger(__name__)


def acf(
    series: Series | None = None,
    lags: ArrayLike = 365,
    bin_method: str = "regular",
    bin_width: float = 0.5,
    max_gap: float = np.inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
    fallback_bin_method: str = "gaussian",
    **kwargs,
) -> Series | DataFrame:
    """Calculate the autocorrelation function for irregular time steps.

    Parameters
    ----------
    series: pandas.Series
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

    See Also
    --------
    pastas.stats.ccf
    statsmodels.api.tsa.acf

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

    >>> acf = ps.stats.acf(series, lags=np.arange(1.0, 366.0, 2.0))

    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")

    if kwargs:
        raise TypeError(f"acf() got unexpected keyword argument '{next(iter(kwargs))}'")

    if series is None:
        raise TypeError("acf() missing required argument: 'series'")

    c = ccf(
        series_x=series,
        series_y=series,
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
    series_x: Series | None = None,
    series_y: Series | None = None,
    lags: ArrayLike = 365,
    bin_method: str = "regular",
    bin_width: float = 0.5,
    max_gap: float = np.inf,
    min_obs: int = 50,
    full_output: bool = False,
    alpha: float = 0.05,
    fallback_bin_method: str = "gaussian",
    **kwargs,
) -> Series | DataFrame:
    """Compute the cross-correlation for irregular time series.

    Parameters
    ----------
    series_x, series_y: pandas.Series
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

    Examples
    --------
    >>> ccf = ps.stats.ccf(series_x, series_y, bin_method="gaussian")
    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series_x` instead of `x`.",
        )
        if series_x is None:
            series_x = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if "y" in kwargs:
        deprecate_args_or_kwargs(
            name="y",
            version="2.4.0",
            reason="Please use `series_y` instead of `y`.",
        )
        if series_y is None:
            series_y = kwargs.pop("y")
        else:
            kwargs.pop("y")

    if kwargs:
        raise TypeError(f"ccf() got unexpected keyword argument '{next(iter(kwargs))}'")

    if series_x is None:
        raise TypeError("ccf() missing required argument: 'series_x'")
    if series_y is None:
        raise TypeError("ccf() missing required argument: 'series_y'")

    # Check if the time series have regular time steps
    if (
        not series_x.index.inferred_freq
        and not series_y.index.inferred_freq
        and bin_method == "regular"
    ):
        msg = (
            f"time series does not have regular time steps, the fallback_bin_method"
            f"'{fallback_bin_method}' is applied"
        )
        logger.warning(msg)
        bin_method = fallback_bin_method

    # prepare the time indices for x and y
    x, t_x, dt_x_mu = _preprocess(series_x, max_gap=max_gap)
    y, t_y, dt_y_mu = _preprocess(series_y, max_gap=max_gap)
    dt_mu = max(dt_x_mu, dt_y_mu)  # The mean time step from both series

    if isinstance(lags, int) and bin_method == "regular":
        lags = np.arange(0, lags + 1, int(dt_mu), dtype=float)
    elif isinstance(lags, int):
        lags = np.arange(0, lags + 1, dtype=float)
    elif isinstance(lags, list):
        lags = np.array(lags, dtype=float)
    elif isinstance(lags, np.ndarray):
        # ensure dtype float otherwise numba will
        # create integer arrays for the results
        lags = lags.astype(float)

    use_numba = options.numba and hasattr(_compute_ccf_gaussian, "signatures")
    if bin_method == "rectangle":
        if use_numba:
            c, b = _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width)
        else:
            c, b = _compute_ccf_rectangle_numpy(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "gaussian":
        if use_numba:
            c, b = _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width)
        else:
            c, b = _compute_ccf_gaussian_numpy(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "regular":
        c, b = _compute_ccf_regular(lags, x, y)
    else:
        raise NotImplementedError

    conf = norm.ppf(1 - alpha / 2.0) / np.sqrt(b)
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


def _preprocess(
    series: Series | None = None, max_gap: float = np.inf, **kwargs
) -> tuple[ArrayLike, ArrayLike, float]:
    """Preprocess the time series."""
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(
            f"_preprocess() got unexpected keyword argument '{next(iter(kwargs))}'"
        )
    if series is None:
        raise TypeError("_preprocess() missing required argument: 'series'")

    x_idx = series.index.to_series().diff().dropna().to_numpy()

    dt = x_idx / Timedelta(1, "D")
    dt_mu = dt[dt < max_gap].mean()  # Deal with big gaps if present
    dt_mu = max(dt_mu, 1)  # Prevent division by zero error
    t = np.concatenate(([0.0], dt.cumsum()))

    # Normalize the values and create numpy arrays
    x_vals = series.to_numpy()  # pandas 3.0: ensure writeable array
    x = (x_vals - x_vals.mean()) / x_vals.std()

    return x, t, dt_mu


@njit(parallel=True, nogil=True, cache=True)  # parallel is controlled by ps.options
def _compute_ccf_rectangle(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute the ccf using rectangular binning."""
    c = np.empty_like(lags)
    b = np.empty_like(lags)
    n = len(t_x)

    is_sorted = True
    for idx in range(1, n):
        if t_x[idx] < t_x[idx - 1] or t_y[idx] < t_y[idx - 1]:
            is_sorted = False
            break

    if is_sorted:
        for k in prange(len(lags)):
            cl = 0.0
            b_sum = 0.0
            lag_k = lags[k]
            tol = 1e-12 * max(1.0, abs(lag_k) + bin_width)
            bw_tol = bin_width + tol
            i_pos = 0
            i_neg = 0
            for j in range(n):
                yj = y[j]
                t_yj = t_y[j]

                # Positive branch: t_x[i] >= t_yj and
                # |t_x[i] - t_yj - lag_k| <= bin_width
                p_low = t_yj + lag_k - bw_tol
                if p_low < t_yj:
                    p_low = t_yj
                p_high = t_yj + lag_k + bw_tol
                if i_pos < j:
                    i_pos = j
                while i_pos < n and t_x[i_pos] < p_low:
                    i_pos += 1
                i = i_pos
                while i < n and t_x[i] <= p_high:
                    if abs(abs(t_x[i] - t_yj) - lag_k) <= bin_width:
                        cl += x[i] * yj
                        b_sum += 1.0
                    i += 1

                # Negative branch: t_x[i] < t_yj (only possible when t_x[j] < t_yj)
                if t_x[j] < t_yj:
                    n_low = t_yj - lag_k - bw_tol
                    n_high = t_yj - lag_k + bw_tol
                    if n_high > t_yj:
                        n_high = t_yj
                    if i_neg < j:
                        i_neg = j
                    while i_neg < n and t_x[i_neg] < n_low:
                        i_neg += 1
                    i = i_neg
                    while i < n and t_x[i] < t_yj and t_x[i] <= n_high:
                        if abs(abs(t_x[i] - t_yj) - lag_k) <= bin_width:
                            cl += x[i] * yj
                            b_sum += 1.0
                        i += 1

            if b_sum == 0.0:
                c[k] = np.nan
                b[k] = 1e-16  # Prevent division by zero error
            else:
                c[k] = cl / b_sum
                b[k] = b_sum
    else:
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
                c[k] = np.nan
                b[k] = 1e-16  # Prevent division by zero error
            else:
                c[k] = cl / b_sum
                b[k] = b_sum
    return c, b


@njit(parallel=True, nogil=True, cache=True)  # parallel is controlled by ps.options
def _compute_ccf_gaussian(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Compute the ccf using Gaussian kernel binning."""
    c = np.empty_like(lags)
    b = np.empty_like(lags)
    n = len(t_x)

    den1 = -2 * bin_width**2  # denominator 1
    den2 = np.sqrt(2 * np.pi * bin_width)  # denominator 2
    six_den2 = 6 * den2  # six std. dev.

    is_sorted = True
    for idx in range(1, n):
        if t_x[idx] < t_x[idx - 1] or t_y[idx] < t_y[idx - 1]:
            is_sorted = False
            break

    if is_sorted:
        for k in prange(len(lags)):
            cl = 0.0
            b_sum = 0.0
            lag_k = lags[k]
            i_start = 0
            for j in range(n):
                t_yj = t_y[j]
                yj = y[j]
                t_low = t_yj + lag_k - six_den2
                t_high = t_yj + lag_k + six_den2
                if i_start < j:
                    i_start = j
                while i_start < n and t_x[i_start] <= t_low:
                    i_start += 1
                i = i_start
                while i < n and t_x[i] < t_high:
                    dtlag = t_x[i] - t_yj - lag_k
                    if abs(dtlag) < six_den2:
                        d = np.exp(dtlag**2 / den1) / den2
                        cl += x[i] * yj * d
                        b_sum += d
                    i += 1
            if b_sum == 0.0:
                c[k] = np.nan
                b[k] = 1e-16  # Prevent division by zero error
            else:
                c[k] = cl / b_sum
                b[k] = b_sum
    else:
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
                        d = np.exp(dtlag**2 / den1) / den2
                        cl += x[i] * yj * d
                        b_sum += d
            if b_sum == 0.0:
                c[k] = np.nan
                b[k] = 1e-16  # Prevent division by zero error
            else:
                c[k] = cl / b_sum
                b[k] = b_sum
    return c, b


def _expand_index_windows(
    lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-row index intervals ``[lo[j], hi[j])`` into flat ``(j, i)`` pairs."""
    counts = hi - lo
    valid = counts > 0
    if not np.any(valid):
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    j_valid = np.nonzero(valid)[0]
    c_valid = counts[valid]
    lo_valid = lo[valid]
    total = int(c_valid.sum())
    j_idx = np.repeat(j_valid, c_valid)
    starts = np.cumsum(c_valid) - c_valid
    i_idx = (
        np.arange(total, dtype=np.intp)
        - np.repeat(starts, c_valid)
        + np.repeat(lo_valid, c_valid)
    )
    return j_idx, i_idx


def _compute_ccf_rectangle_numpy(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Vectorized NumPy implementation of rectangular binning when Numba is disabled."""
    n = len(t_x)
    if n > 1 and (np.any(np.diff(t_x) < 0) or np.any(np.diff(t_y) < 0)):
        py_fn = getattr(_compute_ccf_rectangle, "py_func", _compute_ccf_rectangle)
        return py_fn(lags, t_x, x, t_y, y, bin_width)

    c = np.empty_like(lags, dtype=float)
    b = np.empty_like(lags, dtype=float)
    j_all = np.arange(n, dtype=np.intp)
    has_neg = bool(np.any(t_x < t_y))

    for k in range(len(lags)):
        lag_k = lags[k]
        tol = 1e-12 * max(1.0, abs(lag_k) + bin_width)
        bw_tol = bin_width + tol
        cl = 0.0
        b_sum = 0.0

        # Positive branch: t_x[i] >= t_y[j]
        p_low = np.maximum(t_y, t_y + lag_k - bw_tol)
        p_high = t_y + lag_k + bw_tol
        lo = np.maximum(j_all, np.searchsorted(t_x, p_low, side="left"))
        hi = np.searchsorted(t_x, p_high, side="right")
        j_idx, i_idx = _expand_index_windows(lo, hi)
        if j_idx.size > 0:
            exact = np.abs(np.abs(t_x[i_idx] - t_y[j_idx]) - lag_k) <= bin_width
            if np.any(exact):
                j_ex = j_idx[exact]
                i_ex = i_idx[exact]
                cl += float(np.dot(x[i_ex], y[j_ex]))
                b_sum += float(j_ex.size)

        # Negative branch: t_x[i] < t_y[j]
        if has_neg:
            n_low = t_y - lag_k - bw_tol
            lo_n = np.maximum(j_all, np.searchsorted(t_x, n_low, side="left"))
            hi_n = np.minimum(
                np.searchsorted(t_x, t_y, side="left"),
                np.searchsorted(t_x, t_y - lag_k + bw_tol, side="right"),
            )
            jn, in_ = _expand_index_windows(lo_n, hi_n)
            if jn.size > 0:
                exact = np.abs(np.abs(t_x[in_] - t_y[jn]) - lag_k) <= bin_width
                if np.any(exact):
                    jn_ex = jn[exact]
                    in_ex = in_[exact]
                    cl += float(np.dot(x[in_ex], y[jn_ex]))
                    b_sum += float(jn_ex.size)

        if b_sum == 0.0:
            c[k] = np.nan
            b[k] = 1e-16
        else:
            c[k] = cl / b_sum
            b[k] = b_sum
    return c, b


def _compute_ccf_gaussian_numpy(
    lags: ArrayLike,
    t_x: ArrayLike,
    x: ArrayLike,
    t_y: ArrayLike,
    y: ArrayLike,
    bin_width: float = 0.5,
) -> tuple[ArrayLike, ArrayLike]:
    """Vectorized NumPy implementation of Gaussian binning when Numba is disabled."""
    n = len(t_x)
    if n > 1 and (np.any(np.diff(t_x) < 0) or np.any(np.diff(t_y) < 0)):
        py_fn = getattr(_compute_ccf_gaussian, "py_func", _compute_ccf_gaussian)
        return py_fn(lags, t_x, x, t_y, y, bin_width)

    c = np.empty_like(lags, dtype=float)
    b = np.empty_like(lags, dtype=float)
    j_all = np.arange(n, dtype=np.intp)
    den1 = -2.0 * bin_width**2
    den2 = np.sqrt(2.0 * np.pi * bin_width)
    six_den2 = 6.0 * den2
    # Beyond 9 * bin_width, exp(-40.5) < 3e-18 (negligible in float64)
    eff_cutoff = min(six_den2, 9.0 * bin_width)

    for k in range(len(lags)):
        lag_k = lags[k]
        lo = np.maximum(
            j_all, np.searchsorted(t_x, t_y + lag_k - eff_cutoff, side="right")
        )
        hi = np.searchsorted(t_x, t_y + lag_k + eff_cutoff, side="left")
        j_idx, i_idx = _expand_index_windows(lo, hi)
        if j_idx.size == 0:
            c[k] = np.nan
            b[k] = 1e-16
        else:
            dtlag = t_x[i_idx] - t_y[j_idx] - lag_k
            w = np.exp((dtlag * dtlag) / den1) / den2
            b_sum = float(w.sum())
            if b_sum == 0.0:
                c[k] = np.nan
                b[k] = 1e-16
            else:
                c[k] = float(np.dot(x[i_idx] * y[j_idx], w)) / b_sum
                b[k] = b_sum
    return c, b


def _compute_ccf_regular(
    lags: ArrayLike, x: ArrayLike, y: ArrayLike
) -> tuple[ArrayLike, ArrayLike]:
    c = np.empty_like(lags)
    n = len(x)
    for i in range(len(lags)):
        lag = int(lags[i])
        if lag < n:
            # flip x, y to match numpy/scipy correlate output order
            c[i] = np.corrcoef(y[: n - lag], x[lag:])[0, 1]
        else:
            c[i] = np.nan
    b = n - lags
    b[b <= 0] = 1e-16  # Prevent division by zero error
    return c, b


def mean(
    series: Series | None = None,
    weighted: bool = True,
    max_gap: int = 30,
    **kwargs,
) -> ArrayLike:
    r"""Compute the (weighted) mean of a time series.

    Parameters
    ----------
    series: pandas.Series
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

    .. math::

        \\bar{x} = \\sum_{i=1}^{N} w_i x_i

    where :math:`w_i` are the weights, taken as the time step between observations,
    normalized by the sum of all time steps.
    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(
            f"mean() got unexpected keyword argument '{next(iter(kwargs))}'"
        )
    if series is None:
        raise TypeError("mean() missing required argument: 'series'")

    w = _get_weights(series, weighted=weighted, max_gap=max_gap)
    return np.average(series.to_numpy(), weights=w)


def var(
    series: Series | None = None,
    weighted: bool = True,
    max_gap: int = 30,
    **kwargs,
) -> ArrayLike:
    r"""Compute the (weighted) variance of a time series.

    Parameters
    ----------
    series: pandas.Series
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

    .. math::

        \\sigma_x^2 = \\sum_{i=1}^{N} w_i (x_i - \\bar{x})^2

    where :math:`w_i` are the weights, taken as the time step between observations,
    normalized by the sum of all time steps. Note how weighted mean (:math:`\\bar{
    x}`) is used in this formula.
    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(f"var() got unexpected keyword argument '{next(iter(kwargs))}'")
    if series is None:
        raise TypeError("var() missing required argument: 'series'")

    w = _get_weights(series, weighted=weighted, max_gap=max_gap)
    x = series.to_numpy()
    mu = np.average(x, weights=w)
    sigma = np.sum(x.size / (x.size - 1) * w * (x - mu) ** 2)
    return sigma


def std(
    series: Series | None = None,
    weighted: bool = True,
    max_gap: int = 30,
    **kwargs,
) -> ArrayLike:
    """Compute the (weighted) variance of a time series.

    Parameters
    ----------
    series: pandas.Series
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
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(f"std() got unexpected keyword argument '{next(iter(kwargs))}'")
    if series is None:
        raise TypeError("std() missing required argument: 'series'")

    return np.sqrt(var(series, weighted=weighted, max_gap=max_gap))


def moment(series: Series | None = None, order: int = 0, **kwargs) -> float:
    """Compute the raw moment of an impulse response array.

    Parameters
    ----------
    series : Series
        Impulse response values with a numeric index (float or int)
        representing time steps. A DatetimeIndex is not supported.
    order : int
        The order of the moment to compute (0 through 5).

    Returns
    -------
    moment : float
        The computed raw moment of the impulse response.
    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(
            f"moment() got unexpected keyword argument '{next(iter(kwargs))}'"
        )
    if series is None:
        raise TypeError("moment() missing required argument: 'series'")

    index = series.index
    if isinstance(index, DatetimeIndex):
        raise TypeError(
            "The index of the series must be numeric (float or int) representing "
            "time steps, not a DatetimeIndex. Use a numeric index instead."
        )
    return float(np.sum((index**order) * series.to_numpy()))


# Helper functions


def _get_weights(
    series: Series | None = None,
    weighted: bool = True,
    max_gap: int = 30,
    **kwargs,
) -> ArrayLike:
    """Compute the weights as the time step between observations.

    Parameters
    ----------
    series: pandas.Series
        Series with the values and a DatetimeIndex as an index.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the mean weight. Default
        value is 30 days.
    """
    if "x" in kwargs:
        deprecate_args_or_kwargs(
            name="x",
            version="2.4.0",
            reason="Please use `series` instead of `x`.",
        )
        if series is None:
            series = kwargs.pop("x")
        else:
            kwargs.pop("x")
    if kwargs:
        raise TypeError(
            f"_get_weights() got unexpected keyword argument '{next(iter(kwargs))}'"
        )
    if series is None:
        raise TypeError("_get_weights() missing required argument: 'series'")

    x_index = series.index.to_numpy()
    if weighted:
        w = np.append(0.0, np.diff(x_index) / Timedelta("1D"))
        w[w > max_gap] = max_gap
    else:
        w = np.ones(x_index.size)
    w /= w.sum()
    return w
