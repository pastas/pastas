"""This module contains the objective functions.

"""
from numpy import pi, log
from pandas import DataFrame


class GaussianLikelihood:
    def __init__(self):
        self.nparam = 1

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, name)
        return parameters

    def compute(self, rv, p):
        """

        Parameters
        ----------
        rv
        p

        Returns
        -------

        """
        sigma = p[-1]
        N = rv.size
        ln = -0.5 * N * log(2 * pi * sigma**2) - sum((rv**2) / (2 * sigma**2))
        return ln


class GaussianLikelihoodAr1:
    def __init__(self):
        self.nparam = 2

    def get_init_parameters(self, name: str) -> DataFrame:
        parameters = DataFrame(columns=["initial", "pmin", "pmax", "vary", "name"])
        parameters.loc[name + "_sigma"] = (0.05, 1e-10, 1, True, name)
        parameters.loc[name + "_alpha"] = (0.1, 1e-10, 0.99999, True, name)
        return parameters

    def compute(self, rv, p):
        sigma = p[-2]
        alpha = p[-1]
        N = rv.size
        ln = -(N - 1) / 2 * log(2 * pi * sigma**2) - sum(
            (rv[1:] - alpha * rv[0:-1]) ** 2 / (2 * sigma**2)
        )
        return ln
