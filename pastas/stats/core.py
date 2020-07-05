"""
This module contains core statistical methods.

.. currentmodule:: pastas.stats.core

.. autosummary::
   :nosignatures:
   :toctree: generated/

   acf
   ccf

"""

from numpy import inf, array, unique, exp, sqrt, pi, empty_like
from pandas import Series, Timedelta, DataFrame, TimedeltaIndex

from ..decorators import njit


def acf(x, lags=None, bin_method='rectangle', bin_width=None, max_gap=inf,
        min_obs=10, output="acf", **kwargs):
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
    output: str, optional
        If output is "full", also estimated uncertainties are returned.

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
            max_gap=max_gap, min_obs=min_obs, output=output, **kwargs)
    c.name = "ACF"
    if output == "full":
        return c.rename(columns={"ccf": "acf"})
    else:
        return c


def ccf(x, y, lags=None, bin_method='rectangle', bin_width=None,
        max_gap=inf, min_obs=10, output=None, **kwargs):
    """Method to calculate the cross-correlation function for irregular
    timesteps based on the slotting technique. Different methods (kernels)
    to bin the data are available.

    Parameters
    ----------
    x: pandas.Series
        Pandas Series containing the values to calculate the
        cross-correlation for. The index has to be a Pandas.DatetimeIndex
    lags: array_like, optional
        numpy array containing the lags in days for which the
        cross-correlation is calculated. Default [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 30, 61, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    bin_method: str, optional
        method to determine the type of bin. Options are "rectangle" (default),
        and  "gaussian".
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
    output: str, optional
        If output is "full", also estimated uncertainties are returned.

    Returns
    -------
    c: pandas.Series or pandas.DataFrame
        The Cross-correlation function.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    Examples
    --------
    >>> ccf = ps.stats.ccf(x, y, bin_method="gaussian")

    """
    # prepare the time indices for x and y
    x, t_x, dt_x_min, dt_x_mu = _preprocess(x, max_gap=max_gap,
                                            min_obs=min_obs)
    y, t_y, dt_y_min, dt_y_mu = _preprocess(y, max_gap=max_gap,
                                            min_obs=min_obs)

    dt_mu = max(dt_x_mu, dt_y_mu)
    dt_min = min(dt_x_min, dt_y_min)

    # Default lags in Days, log-scale between 0 and 365.
    if lags is None:
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30, 61,
                90, 120, 150, 180, 210, 240, 270, 300, 330, 365]

    # Remove lags that cannot be determined because lag < dt_min
    lags = array([float(lag) for lag in lags if lag >= dt_min])

    if bin_method == "rectangle":
        if bin_width is None:
            bin_width = 0.5 * dt_mu
        c, b = _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width)
    elif bin_method == "gaussian":
        if bin_width is None:
            bin_width = 0.25 * dt_mu
        c, b = _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width)
    else:
        raise NotImplementedError

    lags = TimedeltaIndex(lags, unit="D", name="Lags")
    dcf = Series(data=c / b, index=lags, name="ccf")

    if output == "full":
        std = 1.96 / sqrt(b - lags.days)
        # std = sqrt((c.cumsum() - dcf.cumsum()) ** 2) / (b - 1)
        dcf = DataFrame(data={"ccf": dcf.values, "stderr": std, "n": b},
                        index=lags, )
    return dcf


def _preprocess(x, max_gap, min_obs):
    """Internal method to preprocess the time series.

    """
    dt = x.index.to_series().diff().values / Timedelta(1, "D")
    dt[0] = 0.0
    dt_mu = dt[dt < max_gap].mean()  # Deal with big gaps if present
    t = dt.cumsum()

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()

    u, i = unique(dt, return_counts=True)
    dt_min = u[Series(i, u).cumsum() >= min_obs][0]

    return x, t, dt_min, dt_mu


@njit
def _compute_ccf_rectangle(lags, t_x, x, t_y, y, bin_width=0.5):
    """Internal numba-optimized method to compute the ccf.

    """
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
        c[k] = cl
        b[k] = b_sum
    return c, b


@njit
def _compute_ccf_gaussian(lags, t_x, x, t_y, y, bin_width=0.5):
    """Internal numba-optimized method to compute the ccf.

    """
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
        c[k] = cl
        b[k] = b_sum

    return c, b