"""The following methods may be used to describe the fit between the model
simulation and the observations.

.. autosummary::
   :nosignatures:
   :toctree: ./generated

   rmse
   sse
   avg_dev
   nse
   evp
   rsq
   bic
   aic

Examples
========
These methods may be used as follows:

>>> ps.stats.rmse(sim, obs)

or

>>> ml.stats.rmse()

"""

from numpy import sqrt, log

__all__ = ["rmse", "sse", "avg_dev", "nse", "evp", "rsq", "bic", "aic"]


def rmse(sim=None, obs=None, res=None, missing="drop"):
    """Root mean squared error.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    res: pandas.Series
        Series with the residual values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    .. math:: rmse = \\sqrt{\\frac{\\sum{r^2}}{n}}

    where :math:`n` is the number of residuals :math:`r`.

    """
    if res is None:
        res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    n = res.size
    return sqrt((res ** 2).sum() / n)


def sse(sim, obs, missing="drop"):
    """Sum of the squares of the error (SSE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The SSE is calculated as follows:

    .. math:: sse = \\sum(r^2)

    Where :math:`r` are the residuals.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    return (res ** 2).sum()


def avg_dev(sim, obs, missing="drop"):
    """Average deviation of the residuals.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    .. math:: avg_{dev} = \\frac{\\sum(r)}{n}

    where :math:`n` is the number of residuals :math:`r`.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    return res.mean()


def nse(sim, obs, missing="drop"):
    """Nash-Sutcliffe coefficient for model fit as defined in [nash_1970]_.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    .. math:: nse = 1 - \\frac{\\sum(h_s-h_o)^2}{\\sum(h_o-\\mu_{h,o})}

    References
    ----------
    .. [nash_1970] Nash, J. E., & Sutcliffe, J. V. (1970). River flow
       forecasting through conceptual models part I-A discussion of
       principles. Journal of hydrology, 10(3), 282-290.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    ns = 1 - (res ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    return ns


def evp(sim, obs, missing="drop"):
    """Explained variance percentage according to [asmuth_2012]_.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    Commonly used goodness-of-fit metric groundwater level models as
    computed in [asmuth_2012]_.

    .. math:: evp = \\frac{\\sigma_h^2 - \\sigma_r^2}{\\sigma_h^2} * 100

    where :math:`\\sigma_h^2` is the variance of the observations and
    :math:`\\sigma_r^2` is the variance of the residuals. The returned value
    is bounded between 0% and 100%.

    References
    ----------
    .. [asmuth_2012] von Asmuth, J., K. Maas, M. Knotters, M. Bierkens,
       M. Bakker, T.N. Olsthoorn, D.G. Cirkel, I. Leunk, F. Schaars, and D.C.
       von Asmuth. 2012. Software for hydrogeologic time series analysis,
       interfacing data with physical insight. Environmental Modelling &
       Software 38: 178–190.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    if obs.var() == 0.0:
        return 100.
    else:
        ev = max(0.0, 100 * (1 - (res.var(ddof=0) / obs.var(ddof=0))))
    return ev


def rsq(sim, obs, nparam=None, missing="drop"):
    """R-squared, possibly adjusted for the number of free parameters.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    .. math:: \\rho_{adj} = 1-  \\frac{n-1}{n-n_{param}}*\\frac{rss}{tss}

    Where n is the number of observations, :math:`n_{param}` the number of
    free parameters, rss the sum of the squared residuals, and tss the total
    sum of squared residuals.

    When nparam is provided, the :math:`\\rho` is
    adjusted for the number of calibration parameters.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    rss = (res ** 2.0).sum()
    tss = ((obs - obs.mean()) ** 2.0).sum()
    if nparam:
        return 1.0 - (obs.size - 1.0) / (obs.size - nparam) * rss / tss
    else:
        return 1.0 - rss / tss


def bic(sim, obs, nparam, missing="drop"):
    """Bayesian Information Criterium (BIC) according to [akaike_1979]_.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The Bayesian Information Criterium (BIC) is calculated as follows:

    .. math:: BIC = -2 log(L) + n_{param} * log(N)

    where :math:`n_{param}` is the number of calibration parameters.

    References
    ----------
    .. [akaike_1979] Akaike, H. (1979). A Bayesian extension of the minimum
       AIC procedure of autoregressive model fitting. Biometrika, 66(2),
       237-242.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    return -2.0 * log((res ** 2.0).sum()) + nparam * log(res.size)


def aic(sim, obs, nparam, missing="drop"):
    """Akaike Information Criterium (AIC) according to [akaike_1974]_.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    .. math:: AIC = -2 log(L) + 2 nparam

    where :math:`n_{param}` is the number of calibration parameters and L is
    the likelihood function for the model.

    References
    ----------
    .. [akaike_1974] Akaike, H. (1974). A new look at the statistical model
       identification. IEEE transactions on automatic control, 19(6), 716-723.

    """
    res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    return -2.0 * log((res ** 2.0).sum()) + 2.0 * nparam
