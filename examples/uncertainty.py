"""This module contains all the methods to obtain information about the
model uncertainty.

"""


class Uncertainty:
    def __init__(self, ml):
        # Save a reference to the model.
        self.ml = ml

    def parameter_sample(self, name=None, parameters=None, n=1000):
        """Method to obtain a set of parameters values based on the optimal
        values and

        Parameters
        ----------
        n: int, optional
            Number of random samples drawn from the bivariate normal
            distribution.
        name: str, optional
            Name of the stressmodel or model component to obtain the
            parameters for.
        parameters:, list of str, optional
            List with the names of the parameters as strings.

        Returns
        -------
        pcov: pandas.DataFrame
            Pandas DataFrame with the parameter samples

        """
        par = self.ml.get_parameters(name=name)

        # Obtain the covariance matrix
        pcov = self.get_pcov(name=name, parameters=parameters)

        par = np.random.multivariate_normal(par, pcov, n)

        return par

    def get_pcov(self, name=None, parameters=None):
        """Internal method to obtain the covariance matrix from the model
        for a specific set of parameters.

        Returns
        -------

        """
        if name:
            params = self.ml.parameters.loc[self.ml.parameters.loc[:,
                                            "name"] == name].index

        pcov = self.ml.fit.pcov.loc[params, params]

        return pcov

    def simulation(self, n):

        params = self.parameter_sample(n=n)

        data = {}

        for i in range(n):
            data[i] = self.ml.simulate(parameters=params[i])

        data = pd.DataFrame(data)

    def block_response(self, name, n=1000):
        params = self.parameter_sample(n=n, name=name)
        data = {}

        for i in range(n):
            data[i] = self.ml.get_block_response(name=name,
                                                 parameters=params[i])

        data = pd.DataFrame(data)
        return data

    def step_response(self, name, n=1000):
        params = self.parameter_sample(n=n, name=name)
        data = {}

        for i in range(n):
            data[i] = self.ml.get_step_response(name=name,
                                                parameters=params[i])

        data = pd.DataFrame(data)
        return data

    def contribution(self, name, n=1000, **kwargs):
        params = self.parameter_sample(n=n, name=name)
        data = {}

        for i in range(n):
            try:
                data[i] = self.ml.get_contribution(name=name,
                                                   parameters=params[i])
            except:
                print(i)
        data = pd.DataFrame(data)
        return data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("First run example.py before running this file!")

ml.uncertainty = Uncertainty(ml)
df = ml.uncertainty.parameter_sample(name="recharge", n=100)
df = ml.uncertainty.block_response(name, 100)

ml.simulate().plot(color="k")
ml.get_block_response("recharge").plot(color="k")
plt.fill_between(df.index, df.quantile(q=0.025, axis=1),
                 df.quantile(q=0.975, axis=1), color="salmon")

name = "Extraction_1"
axes = ml.plots.results()
ml.solve(noise=True)
std = ml.uncertainty.contribution(name=name)  # returns a pandas series?
axes[5].fill_between(std.index, std.quantile(q=0.025, axis=1), std.quantile(
    q=0.975, axis=1), alpha=0.5, color="steelblue")
df = ml.uncertainty.step_response(name, 100)
axes[6].fill_between(df.index, df.quantile(q=0.025, axis=1),
                     df.quantile(q=0.975, axis=1), color="steelblue",
                     alpha=0.5)
# ml.get_contribution(name=name).plot(color="steelblue", x_compat=True)

ml.solve(noise=False)
std = ml.uncertainty.contribution(name=name)  # returns a pandas series?
axes[5].fill_between(std.index, std.quantile(q=0.025, axis=1), std.quantile(
    q=0.975, axis=1), alpha=0.5, color="salmon")
ml.get_contribution(name=name).plot(color="salmon", x_compat=True, ax=axes[5])

df = ml.uncertainty.step_response(name, 100)
axes[6].fill_between(df.index, df.quantile(q=0.025, axis=1),
                     df.quantile(q=0.975, axis=1), color="salmon", alpha=0.5)

plt.savefig("ex_men_uncertainty")
