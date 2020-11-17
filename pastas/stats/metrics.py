"""The following methods may be used to describe the fit between the model
simulation and the observations.

Examples
========
These methods may be used as follows:

>>> ps.stats.rmse(sim, obs)

or

>>> ml.stats.rmse()

"""

from logging import getLogger

from numpy import sqrt, log, ones, nan
from pandas import Timedelta

from pastas.decorators import PastasDeprecationWarning
from pastas.stats.core import mean, var, std

__all__ = ["rmse", "sse", "mae", "nse", "evp", "rsq", "bic", "aic",
           "pearsonr", "kge_2012"]
logger = getLogger(__name__)


# Absolute Error Metrics

def mae(obs=None, sim=None, res=None, missing="drop", weighted=False,
        max_gap=90):
    """Compute the (weighted) Mean Absolute Error (MAE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    The Mean Absolute Error (MAE) between two time series x and y is
    computed as follows:

    .. math:: \\text{MAE} = \\sum_{i=1}^{N} w_i |x_i - y_i|

    where :math:`N` is the number of observations in the observed time series.

    """
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    if weighted:
        w = (res.index[1:] - res.index[:-1]).to_numpy() / Timedelta("1D")
        w[w > max_gap] = w[w <= max_gap].mean()
    else:
        w = ones(res.index.size - 1)
    w /= w.sum()

    return (w * res[1:].abs()).sum()


def rmse(obs=None, sim=None, res=None, missing="drop", weighted=False,
         max_gap=90):
    """Compute the (weighted) Root Mean Squared Error (RMSE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    Computes the Root Mean Squared Error (RMSE) as follows:

    .. math:: \\text{RMSE} = \\sqrt{\\sum_{i=1}^{N} w_i(n_i- \\bar{n})^2}

    where :math:`N` is the number of residuals :math:`n`.

    """
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    if weighted:
        w = (res.index[1:] - res.index[:-1]).to_numpy() / Timedelta("1D")
        w[w > max_gap] = w[w <= max_gap].mean()
    else:
        w = ones(res.index.size - 1)
    w /= w.sum()

    return sqrt((res[1:] ** 2 * w).sum())


def sse(obs=None, sim=None, res=None, missing="drop"):
    """Compute the Sum of the Squared Errors (SSE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The Sum of the Squared Errors (SSE) is calculated as follows:

    .. math:: \\text{SSE} = \\sum(r^2)

    Where :math:`r` are the residuals.

    """
    if res is None:
        res = (sim - obs)

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    return (res ** 2).sum()


@PastasDeprecationWarning
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

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    return res.mean()


# Percentage Error Metrics

def pearsonr(obs, sim, missing="drop", weighted=False, max_gap=90):
    """Compute the (weighted) Pearson correlation (r).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values in the
        observed series. Only "drop" is supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    The Pearson correlation (r) is computed as follows:

    .. math:: r = \\frac{\\sum_{i=1}^{N}w_i (x_i - \\bar{x})(y_i - \\bar{y})}
        {\\sqrt{\\sum_{i=1}^{N} w_i(x_i-\\bar{x})^2 \\sum_{i=1}^{N}
        w_i(y_i-\\bar{y})^2}}

    Where :math:`x` is is observed time series, :math:`y` the simulated
    time series, and :math:`N` the number of observations in the observed
    time series.

    """
    if missing == "drop":
        obs = obs.dropna()

    if weighted:
        w = (obs.index[1:] - obs.index[:-1]).to_numpy() / Timedelta("1D")
        w[w > max_gap] = w[w <= max_gap].mean()
    else:
        w = ones(obs.index.size - 1)

    w /= w.sum()

    sim = sim.reindex(obs.index).dropna()

    # Return nan if the time indices of the sim and obs don't match
    if sim.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    sim = sim[1:] - mean(sim, weighted=weighted, max_gap=max_gap)
    obs = obs[1:] - mean(obs, weighted=weighted, max_gap=max_gap)

    r = (w * sim * obs).sum() / \
        sqrt((w * sim ** 2).sum() * (w * obs ** 2).sum())

    return r


def evp(obs, sim=None, res=None, missing="drop", weighted=False, max_gap=90):
    """Compute the (weighted) Explained Variance Percentage (EVP).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series
        Series with the simulated values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        If weighted is True, the variances are computed using the time
        step between observations as weights. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    Commonly used goodness-of-fit metric groundwater level models as
    computed in [asmuth_2012]_.

    .. math:: \\text{EVP} = \\frac{\\sigma_h^2 - \\sigma_r^2}{\\sigma_h^2}
        * 100

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
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    if obs.var() == 0.0:
        return 100.
    else:
        return max(0.0, (1 - var(res, weighted=weighted, max_gap=max_gap) /
                         var(obs, weighted=weighted, max_gap=max_gap))) * 100


def nse(obs, sim=None, res=None, missing="drop", weighted=False, max_gap=90):
    """Compute the (weighted) Nash-Sutcliffe Efficiency (NSE).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series
        Series with the simulated values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        If weighted is True, the variances are computed using the time
        step between observations as weights. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    .. math:: \\text{NSE} = 1 - \\frac{\\sum(h_s-h_o)^2}{\\sum(h_o-\\mu_{h,o})}

    References
    ----------
    .. [nash_1970] Nash, J. E., & Sutcliffe, J. V. (1970). River flow
       forecasting through conceptual models part I-A discussion of
       principles. Journal of hydrology, 10(3), 282-290.

    """
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    if weighted:
        w = (obs.index[1:] - obs.index[:-1]).to_numpy() / Timedelta("1D")
        w[w > max_gap] = w[w <= max_gap].mean()
    else:
        w = ones(obs.index.size - 1)

    w /= w.sum()
    mu = mean(obs, weighted=weighted, max_gap=max_gap)

    return 1 - (w * res[1:] ** 2).sum() / (w * (obs[1:] - mu) ** 2).sum()


def rsq(obs, sim=None, res=None, missing="drop", nparam=None):
    """Compute R-squared, possibly adjusted for the number of free parameters.

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series
        Series with the simulated values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
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
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    rss = (res ** 2.0).sum()
    tss = ((obs - obs.mean()) ** 2.0).sum()

    if nparam:
        return 1.0 - (obs.size - 1.0) / (obs.size - nparam) * rss / tss
    else:
        return 1.0 - rss / tss


def bic(obs=None, sim=None, res=None, missing="drop", nparam=1):
    """Compute the Bayesian Information Criterium (BIC).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series
        Series with the simulated values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The Bayesian Information Criterium (BIC) [akaike_1979]_ is computed as
    follows:

    .. math:: \\text{BIC} = -2 log(L) + n_{param} * log(N)

    where :math:`n_{param}` is the number of calibration parameters.

    References
    ----------
    .. [akaike_1979] Akaike, H. (1979). A Bayesian extension of the minimum
       AIC procedure of autoregressive model fitting. Biometrika, 66(2),
       237-242.

    """
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    return -2.0 * log((res ** 2.0).sum()) + nparam * log(res.size)


def aic(obs=None, sim=None, res=None, missing="drop", nparam=1):
    """Compute the Akaike Information Criterium (AIC).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series
        Series with the simulated values.
    res: pandas.Series
        Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The Akaike Information Criterium (AIC) [akaike_1974]_ is computed as
    follows:

    .. math:: \\text{AIC} = -2 log(L) + 2 nparam

    where :math:`n_{param}` is the number of calibration parameters and L is
    the likelihood function for the model.

    References
    ----------
    .. [akaike_1974] Akaike, H. (1974). A new look at the statistical model
       identification. IEEE transactions on automatic control, 19(6), 716-723.

    """
    if res is None:
        res = sim - obs

    if missing == "drop":
        res = res.dropna()

    # Return nan if the time indices of the sim and obs don't match
    if res.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    return -2.0 * log((res ** 2.0).sum()) + 2.0 * nparam


# Forecast Error Metrics

def kge_2012(obs, sim, missing="drop", weighted=False, max_gap=90):
    """Compute the (weighted) Kling-Gupta Efficiency (KGE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        mean weight. Default value is 90 days.

    Notes
    -----
    The (weighted) Kling-Gupta Efficiency [kling_2012]_ is computed as follows:

    .. math:: \\text{KGE} = 1 - \\sqrt{(r-1)^2 + (\\beta-1)^2 - (\\gamma-1)^2}

    where :math:`\\beta = \\bar{x} / \\bar{y}` and :math:`\\gamma =
    \\frac{\\bar{\\sigma}_x / \\bar{x}}{\\bar{\\sigma}_y / \\bar{y}}`. If
    weighted equals True, the weighted mean, variance and pearson
    correlation are used.

    References
    ----------
    .. [kling_2012] Kling, H., Fuchs, M., and Paulin, M. (2012). Runoff
      conditions in the upper Danube basin under an ensemble of climate
      change scenarios. Journal of Hydrology, 424-425:264 - 277.

    """
    if missing == "drop":
        obs = obs.dropna()

    sim = sim.reindex(obs.index).dropna()

    # Return nan if the time indices of the sim and obs don't match
    if sim.index.size is 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    r = pearsonr(obs=obs, sim=sim, weighted=weighted, max_gap=max_gap)

    mu_sim = mean(sim, weighted=weighted, max_gap=max_gap)
    mu_obs = mean(obs, weighted=weighted, max_gap=max_gap)

    beta = mu_sim / mu_obs
    gamma = (std(sim, weighted=weighted, max_gap=max_gap) / mu_sim) / \
            (std(obs, weighted=weighted, max_gap=max_gap) / mu_obs)

    kge = 1 - sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return kge
