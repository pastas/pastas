"""This module contains the objective functions that can be used with the pastas
`EmceeSolve` solver.

"""

from numpy import log, pi
from pandas import DataFrame


class GaussianLikelihood:
    """Gaussian likelihood function.

    Notes
    -----
    The Gaussian log-likelihood function is defined as:

    .. math::
        \\log(L) = -\\frac{N}{2}\\log(2\\pi\\sigma^2) +
        \\frac{\\sum_{i=1}^N - \\epsilon_i^2}{2\\sigma^2}

    where :math:`N` is the number of observations, :math:`\\sigma^2` is the variance of
    the residuals, and :math:`\\epsilon_i` is the residual at time :math:`i`. The
    parameter :math:`\\sigma^2` need to be estimated.

    """

    _name = "GaussianLikelihood"

    def __init__(self):
        self.nparam = 1

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get the initial parameters for the log-likelihood function.

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
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"]
        )
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, 0.01, name, "uniform")
        return parameters

    def compute(self, rv, p):
        """Compute the log-likelihood.

        Parameters
        ----------
        rv: array
            Residuals of the model.
        p: array or list
            Parameters of the log-likelihood function.

        Returns
        -------
        ln: float
            Log-likelihood

        """
        sigma = p[-1]
        N = rv.size
        ln = -0.5 * N * log(2 * pi * sigma) + sum(-(rv**2) / (2 * sigma))
        return ln


class GaussianLikelihoodAr1:
    """Gaussian likelihood function with AR1 autocorrelated residuals.

    Notes
    -----
    The Gaussian log-likelihood function with AR1 autocorrelated residual is defined as:

    .. math::
        \\log(L) = -\\frac{N-1}{2}\\log(2\\pi\\sigma^2) +
        \\frac{\\sum_{i=1}^N - (\\epsilon_i - \\phi \\epsilon_{i-\\Delta t})^2}
        {2\\sigma^2}

    where :math:`N` is the number of observations, :math:`\\sigma^2` is the
    variance of the residuals, :math:`\\epsilon_i` is the residual at time
    :math:`i` and :math:`\\mu` is the mean of the residuals. :math:`\\Delta t` is
    the time step between the observations. :math:`\\phi` is the autoregressive
    parameter. The parameters :math:`\\phi` and :math:`\\sigma^2` need to be estimated.

    """

    _name = "GaussianLikelihoodAr1"

    def __init__(self):
        self.nparam = 2

    def get_init_parameters(self, name: str) -> DataFrame:
        """Get the initial parameters for the log-likelihood function.

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
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"]
        )
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, 0.01, name, "uniform")
        parameters.loc[name + "_theta"] = (
            0.5,
            1e-10,
            0.99999,
            True,
            0.2,
            name,
            "uniform",
        )
        return parameters

    def compute(self, rv, p):
        """Compute the log-likelihood.

        Parameters
        ----------
        rv: array
            Residuals of the model.
        p: array or list
            Parameters of the log-likelihood function.

        Returns
        -------
        ln: float
            Log-likelihood.

        """
        sigma = p[-2]
        theta = p[-1]
        N = rv.size
        ln = -(N - 1) / 2 * log(2 * pi * sigma) + sum(
            -((rv[1:] - theta * rv[0:-1]) ** 2) / (2 * sigma)
        )
        return ln
