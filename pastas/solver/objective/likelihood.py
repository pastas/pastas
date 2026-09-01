"""Module containing the likelihood functions used in Pastas for solvers.

These likelihood functions are used in conjunction with Bayesian approaches
(e.g., MCMC) to compute the likelihood of the model given the data.
"""

from abc import ABC, abstractmethod

from numpy import log, pi
from pandas import DataFrame

from ...typing import ArrayLike


class LikelihoodBase(ABC):
    """Abstract base class for likelihood functions.

    This class defines the interface for likelihood functions used in Pastas.
    All likelihood functions should inherit from this class and implement the
    required methods.

    Attributes
    ----------
    _name : str
        Name of the likelihood function.
    nparam : int
        Number of parameters in the likelihood function.
    """

    @abstractmethod
    def get_init_parameters(self, name: str) -> DataFrame:
        """Get initial parameters for the likelihood function.

        Parameters
        ----------
        name: str
            Name of the likelihood function.

        Returns
        -------
        parameters: DataFrame
            Initial parameters for the likelihood function.
        """

    @abstractmethod
    def compute(self, res: ArrayLike, p: ArrayLike) -> float:
        """Compute the log-likelihood.

        Parameters
        ----------
        res: array
            Residuals of the model.
        p: array or list
            Parameters of the likelihood function.

        Returns
        -------
        ln: float
            Log-likelihood.
        """

    @property
    def _name(self) -> str:
        """Name of the likelihood function."""
        return self.__class__.__name__

    @property
    @abstractmethod
    def nparam(self) -> int:
        """Number of parameters in the likelihood function."""


class GaussianLikelihood(LikelihoodBase):
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
    def nparam(self) -> int:
        """Number of parameters in the log-likelihood function."""
        return 1


class GaussianLikelihoodAr1(LikelihoodBase):
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
    def nparam(self) -> int:
        """Number of parameters in the log-likelihood function."""
        return 2
