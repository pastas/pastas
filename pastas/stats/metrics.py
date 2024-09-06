"""The following methods may be used to describe the fit between the model simulation
and the observations.

Examples
========
These methods may be used as follows:

>>> ps.stats.rmse(sim, obs)

or directly from a Pastas model:

>>> ml.stats.rmse()
"""

from logging import captureWarnings, getLogger
from typing import Optional

from numpy import abs as npabs
from numpy import average, log, nan, sqrt
from pandas import Series

from pastas.decorators import PastasDeprecationWarning
from pastas.stats.core import _get_weights, mean, std, var

__all__ = [
    "rmse",
    "sse",
    "mae",
    "nse",
    "evp",
    "rsq",
    "bic",
    "aic",
    "aicc",
    "pearsonr",
    "kge",
]

captureWarnings(True)
logger = getLogger(__name__)
warnings_logger = getLogger("py.warnings")
logger.addHandler(warnings_logger)


# Absolute Error Metrics


def mae(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Mean Absolute Error (MAE).

    Parameters
    ----------
    sim: pandas.Series, optional
        Series with the simulated values.
    obs: pandas.Series, optional
        The Series with the observed values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is True.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.

    Notes
    -----
    The Mean Absolute Error (MAE) between the observed (:math:`y_o`) and simulated
    (:math:`y_s`) time series is computed as follows:

    .. math:: \\text{MAE} = \\sum_{i=1}^{N} w_i |y_s - y_o|

    where :math:`N` is the number of observations in the observed time series.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    w = _get_weights(err, weighted=weighted, max_gap=max_gap)
    return (w * npabs(err.to_numpy())).sum()


def rmse(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Root Mean Squared Error (RMSE).

    Parameters
    ----------
    sim: pandas.Series, optional
        Series with the simulated values.
    obs: pandas.Series, optional
        The Series with the observed values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.

    Notes
    -----
    Computes the Root Mean Squared Error (RMSE) as follows:

    .. math:: \\text{RMSE} = \\sqrt{\\sum_{i=1}^{N} w_i \\epsilon_i^2}

    where :math:`N` is the number of error :math:`\\epsilon`.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    w = _get_weights(err, weighted=weighted, max_gap=max_gap)
    return sqrt((w * err.to_numpy() ** 2).sum())


def sse(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
) -> float:
    """Compute the Sum of the Squared Errors (SSE).

    Parameters
    ----------
    sim: pandas.Series, optional
        Series with the simulated values.
    obs: pandas.Series, optional
        The Series with the observed values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.

    Notes
    -----
    The Sum of the Squared Errors (SSE) is calculated as follows:

    .. math:: \\text{SSE} = \\sum(\\epsilon^2)

    where :math:`\\epsilon` are the errors.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    return (err.to_numpy() ** 2).sum()


# Percentage Error Metrics


def pearsonr(
    obs: Series,
    sim: Series,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Pearson correlation (r).

    Parameters
    ----------
    sim: pandas.Series
        The Series with the simulated values.
    obs: pandas.Series
        The Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values in the observed series. Only
        "drop" is supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for irregular time
        series. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.

    Notes
    -----
    The Pearson correlation (r) is computed as follows:

    .. math:: r = \\frac{\\sum_{i=1}^{N}w_i (y_{o,i} - \\bar{y_o})(y_{s,i} - \\bar{
        y_s})} {\\sqrt{\\sum_{i=1}^{N} w_i(y_{o,i}-\\bar{y_o})^2 \\sum_{i=1}^{N}w_i(
        y_{s,i} -\\bar{y_s})^2}}

    Where :math:`y_o` is observed time series, :math:`y_s` the simulated time series,
    and :math:`N` the number of observations in the observed time series.
    """
    if missing == "drop":
        obs = obs.dropna()

    w = _get_weights(obs, weighted=weighted, max_gap=max_gap)
    sim = sim.reindex(obs.index).dropna().to_numpy()

    # Return nan if the time indices of the sim and obs don't match
    if sim.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    sim = sim - average(sim, weights=w)
    obs = obs.to_numpy() - average(obs.to_numpy(), weights=w)

    r = (w * sim * obs).sum() / sqrt((w * sim**2).sum() * (w * obs**2).sum())

    return r


def evp(
    obs: Series,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Explained Variance Percentage (EVP).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.
    weighted: bool, optional
        If weighted is True, the variances are computed using the time step between
        observations as weights. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.

    Notes
    -----
    Commonly used goodness-of-fit metric groundwater level models as computed in
    :cite:t:`von_asmuth_groundwater_2012`.

    .. math:: \\text{EVP} = \\frac{\\sigma_h^2 - \\sigma_r^2}{\\sigma_h^2}
        * 100

    where :math:`\\sigma_h^2` is the variance of the observations and
    :math:`\\sigma_r^2` is the variance of the errors. The returned value is
    bounded between 0% and 100%.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    if obs.var() == 0.0:
        return 100.0
    else:
        return (
            max(
                0.0,
                (
                    1
                    - var(err, weighted=weighted, max_gap=max_gap)
                    / var(obs, weighted=weighted, max_gap=max_gap)
                ),
            )
            * 100
        )


def nse(
    obs: Series,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Nash-Sutcliffe Efficiency (NSE).

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.
    weighted: bool, optional
        If weighted is True, the variances are computed using the time step between
        observations as weights. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.

    Notes
    -----
    NSE computed according to :cite:t:`nash_river_1970`

    .. math:: \\text{NSE} = 1 - \\frac{\\sum(h_s-h_o)^2}{\\sum(h_o-\\mu_{h,o})}
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    w = _get_weights(err, weighted=weighted, max_gap=max_gap)
    mu = average(obs.to_numpy(), weights=w)

    return 1 - (w * err.to_numpy() ** 2).sum() / (w * (obs.to_numpy() - mu) ** 2).sum()


def rsq(
    obs: Series,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
    nparam: Optional[int] = None,
) -> float:
    """Compute R-squared, possibly adjusted for the number of free parameters.

    Parameters
    ----------
    obs: pandas.Series
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.
    weighted: bool, optional
        If weighted is True, the variances are computed using the time step between
        observations as weights. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the weights.
        All time steps larger than max_gap are replace with the max_gap value.
        Default value is 30 days.
    nparam: int, optional
        number of calibrated parameters.

    Notes
    -----
    .. math:: \\rho_{adj} = 1-  \\frac{n-1}{n-n_{param}}*\\frac{rss}{tss}

    Where n is the number of observations, :math:`n_{param}` the number of free
    parameters, rss the sum of the squared errors, and tss the total sum of
    squared errors.

    When nparam is provided, the :math:`\\rho` is adjusted for the number of
    calibration parameters.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    w = _get_weights(err, weighted=weighted, max_gap=max_gap)
    if len(w) != obs.index.size:
        raise ValueError(
            "Weights and observations time series have different lengths! "
            "Check observation and simulation time series."
        )
    mu = average(obs.to_numpy(), weights=w)
    rss = (w * err.to_numpy() ** 2.0).sum()
    tss = (w * (obs.to_numpy() - mu) ** 2.0).sum()

    if nparam:
        return 1.0 - (obs.size - 1.0) / (obs.size - nparam) * rss / tss
    else:
        return 1.0 - rss / tss


def bic(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    nparam: int = 1,
) -> float:
    """Compute the Bayesian Information Criterium (BIC).

    Parameters
    ----------
    obs: pandas.Series, optional
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.

    Notes
    -----
    The Bayesian Information Criterium (BIC) :cite:p:`akaike_bayesian_1979` is
    computed as follows:

    .. math:: \\text{BIC} = -2 log(L) + n_{param} * log(N)

    where :math:`n_{param}` is the number of calibration parameters.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    n = err.index.size

    return n * log((err.to_numpy() ** 2.0).sum() / n) + nparam * log(n)


def aic(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    nparam: int = 1,
) -> float:
    """Compute the Akaike Information Criterium (AIC).

    Parameters
    ----------
    obs: pandas.Series, optional
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored. Note that the
        residuals must be computed as `obs - sim` here.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----
    The Akaike Information Criterium (AIC) :cite:p:`akaike_new_1974` is computed as
    follows:

    .. math:: \\text{AIC} = -2 log(L) + 2 nparam

    where :math:`n_{param}` is the number of calibration parameters and L is
    the likelihood function for the model. In the case of ordinary least
    squares:

    .. math:: log(L) = - (nobs / 2) * log(RSS / -nobs)

    where RSS denotes the residual sum of squares and nobs the number of
    observations.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    # Return nan if the time indices of the sim and obs don't match
    if err.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    n = err.index.size

    return n * log((err.to_numpy() ** 2.0).sum() / n) + 2.0 * nparam


def aicc(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
    nparam: int = 1,
) -> float:
    """Compute the Akaike Information Criterium with second order
    bias correction for the number of observations (AICc)

    Parameters
    ----------
    obs: pandas.Series, optional
        Series with the observed values.
    sim: pandas.Series, optional
        The Series with the simulated values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals
        are provided, the sim and obs arguments are ignored. Note that the
        residuals must be computed as `obs - sim` here.
    nparam: int, optional
        number of calibrated parameters.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.

    Notes
    -----

    The corrected Akaike Information Criterium (AICc)
    :cite:p:`suguria_aicc_1978` is computed as follows:

    .. math:: \\text{AIC} = -2 log(L) + 2 nparam - (2 nparam (nparam + 1) / (nobs - nparam - 1))

    where :math:`n_{param}` is the number of calibration parameters, nobs is
    the number of observations and L is the likelihood function for the model.
    In the case of ordinary least squares:

    .. math:: log(L) = - (nobs / 2) * log(RSS / -nobs)

    where RSS denotes the residual sum of squares.
    """
    err = _compute_err(obs=obs, sim=sim, res=res, missing=missing)

    n = err.index.size

    c_term = (2 * nparam * (nparam + 1)) / (n - nparam - 1)
    return aic(res=-err, nparam=nparam) + c_term


# Forecast Error Metrics


def kge(
    obs: Series,
    sim: Series,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
    modified: bool = False,
) -> float:
    """Compute the (weighted) Kling-Gupta Efficiency (KGE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        The Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        max_gap value. Default value is 30 days.
    modified: bool, optional
        Use the modified KGE as proposed by :cite:t:`kling_runoff_2012`.
        According to the article this ensures that the bias and variability
        ratios are not cross-correlated, which otherwise may occur when inputs
        are biased.

    Notes
    -----
    The (weighted) Kling-Gupta Efficiency :cite:t:`kling_runoff_2012` is
    computed as follows:

    .. math:: \\text{KGE} = 1 - \\sqrt{(r-1)^2 + (\\beta-1)^2 - (\\gamma-1)^2}

    where :math:`\\beta = \\bar{x} / \\bar{y}` and :math:`\\gamma =
    \\frac{\\bar{\\sigma}_x}{\\bar{\\sigma}_y}`. If modified equals True,
    :math:`\\gamma = \\frac{\\bar{\\sigma}_x / \\bar{x}}{\\bar{\\sigma}_y /
    \\bar{y}}`. If weighted equals True, the weighted mean, variance and
    pearson correlation are used.
    """
    if missing == "drop":
        obs = obs.dropna()

    sim = sim.reindex(obs.index).dropna()

    # Return nan if the time indices of the sim and obs don't match
    if sim.index.size == 0:
        logger.warning("Time indices of the sim and obs don't match.")
        return nan

    r = pearsonr(obs=obs, sim=sim, weighted=weighted, max_gap=max_gap)

    mu_sim = mean(sim, weighted=weighted, max_gap=max_gap)
    mu_obs = mean(obs, weighted=weighted, max_gap=max_gap)

    beta = mu_sim / mu_obs
    if modified:
        gamma = (std(sim, weighted=weighted, max_gap=max_gap) / mu_sim) / (
            std(obs, weighted=weighted, max_gap=max_gap) / mu_obs
        )
    else:
        gamma = std(sim, weighted=weighted, max_gap=max_gap) / std(
            obs, weighted=weighted, max_gap=max_gap
        )

    kge = 1 - sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return kge


@PastasDeprecationWarning(
    remove_version="2.0",
    reason="""This function `kge_2012` will be deprecated in Pastas version 2.0. Please
    use `pastas.stats.kge(modified=True)` to get the same outcome.""",
)
def kge_2012(
    obs: Series,
    sim: Series,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Kling-Gupta Efficiency (KGE).

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        The Series with the observed values.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is
        supported now.
    weighted: bool, optional
        Weight the values by the normalized time step to account for
        irregular time series. Default is False.
    max_gap: int, optional
        maximum allowed gap period in days to use for the computation of the
        weights. All time steps larger than max_gap are replace with the
        max_gap value. Default value is 30 days.

    Notes
    -----
    The (weighted) Kling-Gupta Efficiency :cite:t:`kling_runoff_2012` is
    computed as follows:

    .. math:: \\text{KGE} = 1 - \\sqrt{(r-1)^2 + (\\beta-1)^2 - (\\gamma-1)^2}

    where :math:`\\beta = \\bar{x} / \\bar{y}` and :math:`\\gamma =
    \\frac{\\bar{\\sigma}_x / \\bar{x}}{\\bar{\\sigma}_y / \\bar{y}}`. If
    weighted equals True, the weighted mean, variance and pearson
    correlation are used.
    """
    return kge(
        obs=obs,
        sim=sim,
        missing=missing,
        weighted=weighted,
        max_gap=max_gap,
        modified=True,
    )


def _compute_err(
    obs: Optional[Series] = None,
    sim: Optional[Series] = None,
    res: Optional[Series] = None,
    missing: str = "drop",
):
    """

    Parameters
    ----------
    Parameters
    ----------
    sim: pandas.Series, optional
        Series with the simulated values.
    obs: pandas.Series, optional
        The Series with the observed values.
    res: pandas.Series, optional
        The Series with the residual values. If time series for the residuals are
        provided, the sim and obs arguments are ignored. Note that the residuals
        must be computed as `obs - sim` here.
    missing: str, optional
        string with the rule to deal with missing values. Only "drop" is supported now.

    Returns
    -------
    err: pandas.Series
        The pandas.Series with the errors, computed as

    """
    if (obs is not None) and (sim is not None):
        err = sim.subtract(obs)
    elif res is not None:
        err = -res
    else:
        msg = (
            "Either the residuals, or the simulation and observations have to be "
            "provided. Please provide one of these two input options."
        )
        logger.error(msg)
        raise ValueError(msg)

    if missing == "drop":
        err = err.dropna()

    return err
