"""Module containing the objective function for solvers.

This contains the misfit which calculates the calculate residuals or noise.

This module also contains the likelihood function used in Pastas for solvers
using Bayesian approaches (e.g., MCMC) to compute the likelihood of the model
given the data.
"""

from collections.abc import Callable

from numpy import log, pi
from pandas import DataFrame, Series

from pastas.typing import ArrayLike, Model


def misfit(
    model: Model,
    p: ArrayLike,
    noise: bool,
    weights: Series | None = None,
    callback: Callable | None = None,
    returnseparate: bool = False,
) -> ArrayLike | tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Shared objective function for solvers to calculate residuals or noise.

    Parameters
    ----------
    model: object
        The model instance containing residuals and noise methods.
    p: np.ndarray
        Array of parameter values.
    noise: bool
        If True, minimizes the sum of squared noise computed by the NoiseModel.
    weights: pandas.Series, optional
        Weights to scale the residuals or noise.
    callback: Callable, optional
        Function to call after each iteration.
    returnseparate: bool, optional
        If True, returns residuals, noise, and noise weights separately.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray, np.ndarray]
        The calculated residuals or noise, optionally with separate components.
    """
    # Get the residuals or the noise
    res = model.residuals(p)
    if noise:
        res = model.noise(p=p, res=res) * model._noise_weights(p=p, res=res)

    # Apply weights if provided
    if weights is not None:
        weights = weights.reindex(res.index)
        weights.fillna(1.0, inplace=True)
        res = res.multiply(weights)

    # Call the callback function if provided
    if callback is not None:
        callback(p)

    # Return separate components if requested
    if returnseparate:
        return (
            res.to_numpy(copy=True),
            model.noise(p=p, res=res).to_numpy(copy=True),
            model._noise_weights(p=p, res=res).to_numpy(copy=True),
        )

    return res.to_numpy(copy=True)


class GaussianLikelihood:
    r"""Gaussian likelihood function for homoscedastic, uncorrelated errors.

    Notes
    -----
    The Gaussian log-likelihood function :cite:p:`smith_modeling_2015` is defined as:

    .. math::

        \\log(L) = -\\frac{N}{2}\\log(2\\pi\\sigma^2) -
        \\frac{\\sum_{t=1}^N \\epsilon_t^2}{2\\sigma^2}

    where :math:`N` is the number of observations, :math:`\\sigma^2` is the variance of
    the residuals, and :math:`\\epsilon_t` is the residual at time :math:`t`. The
    parameter :math:`\\sigma^2` needs to be estimated.

    The current implementation is valid for equidistant time series only.

    """

    def __init__(self) -> None:
        pass

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters for the log-likelihood function.

        Parameters
        ----------
        name: str
            Name of the log-likelihood function.

        Returns
        -------
        parameters: DataFrame
            Initial parameters for the log-likelihood function.

        """
        parameters = DataFrame(
            [(0.05, 1e-10, 1.0, True, name, 1.0, "norm")],
            columns=[
                "initial",
                "pmin",
                "pmax",
                "vary",
                "name",
                "sigma",
                "dist",
            ],
            index=[name + "_var"],
        )
        return parameters

    def compute(self, res: ArrayLike, p: ArrayLike) -> float:
        """Compute the log-likelihood.

        Parameters
        ----------
        res: array
            Residuals of the model.
        p: array or list
            Parameters of the log-likelihood function.

        Returns
        -------
        ln: float
            Log-likelihood

        """
        var = p[-1]
        N = len(res)
        ln = -0.5 * N * log(2 * pi * var) + sum(-(res**2) / (2 * var))
        return ln

    @property
    def _name(self) -> str:
        """Name of the log-likelihood function."""
        return self.__class__.__name__

    @property
    def nparam(self) -> int:
        """Number of parameters in the log-likelihood function."""
        return 1


class GaussianLikelihoodAr1:
    r"""Gaussian likelihood function for homoscedastic, autocorrelated residuals.

    Notes
    -----
    The Gaussian log-likelihood function with AR1 autocorrelated residuals
    :cite:p:`smith_modeling_2015` is defined as:

    .. math::

        \\log(L) = -\\frac{N-1}{2}\\log(2\\pi\\sigma^2) -
         \\frac{\\sum_{t=1}^N(\\epsilon_t - \\phi \\epsilon_{t-\\Delta t})^2}
        {2\\sigma^2}

    where :math:`N` is the number of observations, :math:`\\sigma^2` is the
    variance of the residuals, :math:`\\epsilon_t` is the residual at time
    :math:`t`. :math:`\\Delta t` is the time step between the observations.
    :math:`\\phi` is the autoregressive parameter. The parameters :math:`\\phi` and
    :math:`\\sigma^2` need to be estimated.

    The current implementation is valid for equidistant time series only.

    """

    def __init__(self) -> None:
        pass

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters for the log-likelihood function.

        Parameters
        ----------
        name: str
            Name of the log-likelihood function.

        Returns
        -------
        parameters: DataFrame
            Initial parameters for the log-likelihood function.

        """
        return DataFrame(
            [
                (0.05, 1e-10, 1.0, True, name, 1.0, "norm"),
                (0.5, 1e-10, 0.99999, True, name, 1.0, "norm"),
            ],
            columns=[
                "initial",
                "pmin",
                "pmax",
                "vary",
                "name",
                "sigma",
                "dist",
            ],
            index=[name + "_var", name + "_phi"],
        )

    def compute(self, res: ArrayLike, p: ArrayLike) -> float:
        """Compute the log-likelihood.

        Parameters
        ----------
        res: array
            Residuals of the model.
        p: array or list
            Parameters of the log-likelihood function.

        Returns
        -------
        ln: float
            Log-likelihood.

        """
        var = p[-2]
        phi = p[-1]
        N = len(res)
        ln = -(N - 1) / 2 * log(2 * pi * var) + sum(
            -((res[1:] - phi * res[0:-1]) ** 2) / (2 * var)
        )
        return ln

    @property
    def _name(self) -> str:
        """Name of the log-likelihood function."""
        return self.__class__.__name__

    @property
    def nparam(self) -> int:
        """Number of parameters in the log-likelihood function."""
        return 2
