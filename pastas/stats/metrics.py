"""The following methods may be used to describe the fit between the model simulation
and the observations.

Examples
========
These methods may be used as follows:

>>> ps.stats.rmse(sim, obs)

or directly from a Pastas model:

>>> ml.stats.rmse()
"""

from logging import getLogger

from numpy import abs as npabs
from numpy import average, log, nan, sqrt
from pandas import DataFrame, Series

from pastas.decorators import PastasDeprecationWarning
from pastas.stats.core import _get_weights, mean, std, var

__all__ = [
    "rmse",
    "sse",
    "mae",
    "nse",
    "nnse",
    "evp",
    "rsq",
    "bic",
    "aic",
    "aicc",
    "pearsonr",
    "kge",
    "picp",
]

logger = getLogger(__name__)

# Absolute Error Metrics


def mae(
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    sim: Series | None = None,
    res: Series | None = None,
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
    sim: Series | None = None,
    res: Series | None = None,
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


def nnse(
    obs: Series,
    sim: Series | None = None,
    res: Series | None = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
) -> float:
    """Compute the (weighted) Normalized Nash-Sutcliffe Efficiency (NNSE).

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
    NNSE computed according to :cite:t:`nash_normalized_2006`

    .. math:: \\text{NNSE} = 1 / (2 - NSE)

    This metric normalizes the NSE between ~0 and 1 instead of -infinity and 1.
    So the optimal value for NNSE is 1, same as the NSE. However, an NNSE value
    of 0.5 corresponds to an NSE of 0, while the worst possible NNSE value is ~0.
    """
    nnse = 1 / (
        2
        - nse(
            obs=obs,
            sim=sim,
            res=res,
            missing=missing,
            weighted=weighted,
            max_gap=max_gap,
        )
    )
    return nnse


def rsq(
    obs: Series,
    sim: Series | None = None,
    res: Series | None = None,
    missing: str = "drop",
    weighted: bool = False,
    max_gap: int = 30,
    nparam: int | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
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
    obs: Series | None = None,
    sim: Series | None = None,
    res: Series | None = None,
    missing: str = "drop",
):
    """
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


# Prediction Interval Metrics


def picp(obs: Series, bounds: DataFrame):
    """Compute the prediction interval coverage probability (PICP).

    Parameters
    ----------
    obs : pandas.Series
        Pandas Series with the measured time series and a DateTimeIndex.
    bounds : DataFrame
        DataFrame with the lower (first column) and upper (second columns) bounds of the
        prediction intervals.

    Notes
    -----
    The Prediction Interval Coverage Probability (PICP) is computed as follows:

    .. math:: PICP = \\frac{1}{N} \\sum_{i=1}^N a_i,
        a_i =
        \\begin{cases}
             1 & \\text{if} h_i \\text{in} [\\hat{h_i}^L, \\hat{h_i}^U], \\
             0 & \\text{otherwise}
        \\end{cases}

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pastas.stats import picp
    >>> obs = pd.Series(np.random.rand(100),
                        index=pd.date_range("2000-01-01", periods=100))
    >>> bounds = pd.DataFrame(np.random.rand(100, 2), index=obs.index)
    >>> picp(obs, bounds)

    """
    # Check if the DateTimeIndex of the observations and the bounds match
    if not obs.index.isin(bounds.index).all():
        msg = "The DateTimeIndex of the observations and the bounds do not match."
        logger.error(msg)
        raise ValueError(msg)

    if not obs.index.equals(bounds.index):
        bounds = bounds.loc[obs.index, :]

    # Determine if the observations are within the prediction intervals
    a = (obs >= bounds.iloc[:, 0]) & (obs <= bounds.iloc[:, 1])
    return a.mean()
