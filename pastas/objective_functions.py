"""This module contains the objective functions that can be used with the pastas
`EmceeSolve` solver.

"""

from numpy import log, pi
from pandas import DataFrame


class GaussianLikelihood:
    """Gaussian likelihood function for homoscedastic, uncorrelated errors.

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
            [(0.05, 1e-10, 1.0, True, 0.01, name, "uniform")],
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"],
            index=[name + "_var"],
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
            Log-likelihood

        """
        var = p[-1]
        N = rv.size
        ln = -0.5 * N * log(2 * pi * var) + sum(-(rv**2) / (2 * var))
        return ln


class GaussianLikelihoodAr1:
    """Gaussian likelihood function for homoscedastic, autocorrelated residuals.

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
        return DataFrame(
            [
                (0.05, 1e-10, 1.0, True, 0.01, name, "uniform"),
                (0.5, 1e-10, 0.99999, True, 0.2, name, "uniform"),
            ],
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"],
            index=[name + "_var", name + "_phi"],
        )

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
        var = p[-2]
        phi = p[-1]
        N = rv.size
        ln = -(N - 1) / 2 * log(2 * pi * var) + sum(
            -((rv[1:] - phi * rv[0:-1]) ** 2) / (2 * var)
        )
        return ln
