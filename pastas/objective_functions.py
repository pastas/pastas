"""This module contains the objective functions.

"""
from numpy import pi, log, nan
from pandas import DataFrame


class GaussianLikelihood:
    """ Gaussian likelihood function.

    """
    def __init__(self):
        self.nparam = 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"]
        )
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, nan, name, "uniform")
        return parameters

    def compute(self, rv, p):
        """Compute the log-likelihood.

        Parameters
        ----------
        rv: array
            Residuals of the model
        p: array or list
            Parameters of the noise model

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
    """

    """
    def __init__(self):
        self.nparam = 2

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(
            columns=["initial", "pmin", "pmax", "vary", "stderr", "name", "dist"]
        )
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, 0.01, name, "uniform")
        parameters.loc[name + "_alpha"] = (
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
        sigma = p[-2]
        alpha = p[-1]
        N = rv.size
        ln = -(N - 1) / 2 * log(2 * pi * sigma) + sum(
            -((rv[1:] - alpha * rv[0:-1]) ** 2) / (2 * sigma)
        )
        return ln
