"""Statistics for time series models.
Statistics can be calculated for the following time series:
- Observation series
- Simulated series
- Residual series
- Innovation series
Each of these series can be obtained through their individual (private) get
method for a specific time frame.
two different types of statistics are provided: model statistics and
descriptive statistics for each series.
Usage
-----
>>> ml.stats.summary()
                                     Value
Statistic
Pearson R^2                       0.874113
Root mean squared error           0.432442
Bayesian Information Criterion  113.809120
Average Deviation                 0.335966
Explained variance percentage    72.701968
Akaike InformationCriterion      25.327385
TODO
----
- ACF for irregular timesteps
- PACF for irregular timestep
- Nash-Sutcliffe
- portmanteau test (ljung-Box & Box-Pierce)
"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot
from statsmodels.tsa.stattools import acf, pacf


class Statistics(object):
    def __init__(self, ml):
        """
        To obtain a list of all statistics that are
        included type:

        >>> print(ml.stats.ops)

        ml: Pastas Model
            ml is a time series Model that is calibrated.
        """
        # Save a reference to the model.
        self.ml = ml
        # Save all statistics that can be calculated.

        self.ops = {'evp': 'Explained variance percentage',
                    'rmse': 'Root mean squared error',
                    'rmsi': 'Root mean squared innovation',
                    'sse': 'Sum of squares of the error',
                    'avg_dev': 'Average Deviation',
                    'rsq': 'Pearson R^2',
                    'rsq_adj': 'Adjusted Pearson R^2',
                    'bic': 'Bayesian Information Criterion',
                    'aic': 'Akaike Information Criterion'}

    def __repr__(self):
        msg = """This module contains all the statistical functions that are
included in Pastas. To obtain a list of all statistics that are included type:

    >>> print(ml.stats.ops)"""
        return msg

    # The statistical functions

    def rmse(self, tmin=None, tmax=None):
        """Root mean squared error of the residuals.

        Notes
        -----
        .. math:: rmse = sqrt(sum(residuals**2) / N)

        where N is the number of residuals.
        """
        res = self.ml.get_residuals(tmin, tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def rmsi(self, tmin=None, tmax=None):
        """Root mean squared error of the innovations.

        Notes
        -----
        .. math:: rmsi = sqrt(sum(innovations**2) / N)

        where N is the number of innovations.
        """
        res = self.ml.get_innovations(tmin, tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def sse(self, tmin=None, tmax=None):
        """Sum of the squares of the error (SSE)

        Notes
        -----
        The SSE is calculated as follows:

        .. math:: SSE = sum(E ** 2)

        Where E is an array of the residual series.

        """
        res = self.ml.get_residuals(tmin, tmax)
        return sum(res ** 2)

    def avg_dev(self, tmin=None, tmax=None):
        """Average deviation of the residuals.

        Notes
        -----
        .. math:: avg_dev = sum(E) / N

        Where N is the number of the residuals.

        """
        res = self.ml.get_residuals(tmin, tmax)
        return res.mean()

    def evp(self, tmin=None, tmax=None):
        """Explained variance percentage.

        Notes
        -----
        Commonly used statistic in time series models of groundwater level.
        It has to be noted that a high EVP value does not necessarily indicate
        a good time series model.

        .. math:: evp = (var(h) - var(res)) / var(h) * 100%

        """
        res = self.ml.get_residuals(tmin, tmax)
        obs = self.ml.get_observations(tmin, tmax)
        return (np.var(obs) - np.var(res)) / np.var(obs) * 100.0

    def rsq(self, tmin=None, tmax=None):
        """Correlation between observed and simulated series.

        Notes
        -----
        For the calculation of this statistic the corrcoef method from numpy
        is used.

        >>> np.corrcoef(sim, obs)[0, 1]

        Please refer to the Numpy Docs:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html#numpy.corrcoef

        """
        sim = self.ml.get_simulation(tmin, tmax)
        obs = self.ml.get_observations(tmin, tmax)
        sim = sim[obs.index]  # Make sure to correlate the same in time.
        return np.corrcoef(sim, obs)[0, 1]

    def rsq_adj(self, tmin=None, tmax=None):
        """R-squared Adjusted for the number of free parameters.

        Notes
        -----
        .. math:: R_{corrected} = 1 - (n-1) / (n-N_param) * RSS/TSS

        Where:
            RSS = sum of the squared residuals.
            TSS = ??
            N = Number of observations
            N_Param = Number of free parameters
        """

        obs = self.ml.get_observations(tmin, tmax)
        res = self.ml.get_residuals(tmin, tmax)
        N = obs.size

        RSS = sum(res ** 2.0)
        TSS = sum((obs - obs.mean()) ** 2.0)
        return 1.0 - (N - 1.0) / (N - self.ml.nparam) * RSS / TSS

    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium.

        Notes
        -----
        The Bayesian Information Criterium is calculated as follows:

        .. math:: BIC = -2 log(L) + nparam * log(N)

        Where:
            nparam : Number of free parameters
        """
        innovations = self.ml.get_innovations(tmin, tmax)
        n = innovations.size
        nparam = len(self.ml.parameters[self.ml.parameters.vary == True])
        bic = -2.0 * np.log(sum(innovations ** 2.0)) + nparam * np.log(n)
        return bic

    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium (AIC).

        Notes
        -----
        .. math:: AIC = -2 log(L) + 2 nparam

        Where
            nparam = Number of free parameters
            L = likelihood function for the model.
        """
        innovations = self.ml.get_innovations(tmin, tmax)
        nparam = len(self.ml.parameters[self.ml.parameters.vary == True])
        aic = -2.0 * np.log(sum(innovations ** 2.0)) + 2.0 * nparam
        return aic

    def acf(self, tmin=None, tmax=None, nlags=20):
        """Autocorrelation function.

        Notes
        -----
        For the autocorrelation function the acf method from Statsmodels
        package is used untill an alternative is found. However, please be
        aware that this can lead to incorrect values for irregular time steps.

        Please refer to the Statsmodels docs:
        http://statsmodels.sourceforge.net/devel/_modules/statsmodels/tsa/stattools.html#acf

        TODO: Compute autocorrelation for irregulat time steps.
        """
        innovations = self.ml.get_innovations(tmin, tmax)
        return acf(innovations, nlags=nlags)

    def pacf(self, tmin=None, tmax=None, nlags=20):
        """Partial autocorrelation function.

        Notes
        -----
        For the partial autocorrelation function the acf method from
        Statsmodels package is used untill an alternative is found. However,
        please be aware that this can lead to incorrect values for irregular
        time steps.

        Please refer to the Statsmodels docs:
        http://statsmodels.sourceforge.net/devel/_modules/statsmodels/tsa/stattools.html#pacf

        TODO: Compute  partial autocorrelation for irregulat time steps.
        """
        innovations = self.ml.get_innovations(tmin, tmax)
        return pacf(innovations, nlags=nlags)

    def all(self, tmin=None, tmax=None):
        """Returns a dictionary with all the statistics.

        Parameters
        ----------
        tmin: str
        tmax: str

        Returns
        -------
        stats: pd.DataFrame
            Dataframe with all possible statistics

        """

        stats = pd.DataFrame(columns=['Value'])
        for k in self.ops.keys():
            stats.loc[k] = (getattr(self, k)(tmin, tmax))

        return stats

    # def GHG(self, tmin=None, tmax=None, series='oseries'):
    #     """GHG: Gemiddeld Hoog Grondwater (in Dutch)
    #
    #     3 maximum groundwater level observations for each year divided by 3 times
    #     the number of years.
    #
    #     Parameters
    #     ----------
    #     series: Optional[str]
    #         string for the series to calculate the statistic for. Supported
    #         options are: 'oseries'.
    #
    #     """
    #     if series == 'oseries':
    #         x = []
    #         oseries = self.ml.get_observations(tmin, tmax)
    #         for year in np.unique(oseries.index.year):
    #             x.append(oseries['%i' % year].sort_values(ascending=False,
    #                                                       inplace=False)[
    #                      0:3].values)
    #         return np.mean(np.array(x))
    #
    # def GLG(self, tmin=None, tmax=None, series='oseries'):
    #     """GLG: Gemiddeld Laag Grondwater (in Dutch)
    #
    #     3 minimum groundwater level observations for each year divided by 3 times
    #     the number of years.
    #
    #     Parameters
    #     ----------
    #     series: Optional[str]
    #         string for the series to calculate the statistic for. Supported
    #         options are: 'oseries'.
    #
    #     """
    #     if series == 'oseries':
    #         x = []
    #         oseries = self.ml.get_observations(tmin, tmax)
    #         for year in np.unique(oseries.index.year):
    #             x.append(oseries['%i' % year].sort_values(ascending=True,
    #                                                       inplace=False)[
    #                      0:3].values)
    #         return np.mean(np.array(x))

    def descriptive(self, tmin=None, tmax=None):
        """Returns the descriptive statistics for all time series.

        """
        return print("This method is currently not supported. Please use:"
                     ">>> ml.get_simulations().describe() to make use of the "
                     "built-in Pandas methods.")
        #series = self.__getallseries__(tmin, tmax)
        #series.describe()

    def plot_diagnostics(self, tmin=None, tmax=None):
        innovations = self.ml.get_innovations(tmin, tmax)

        plt.figure()
        gs = plt.GridSpec(2, 3, wspace=0.2)

        plt.subplot(gs[0, :2])
        plt.title('Autocorrelation')
        # plt.axhline(0.2, '--')
        plt.stem(self.acf())

        plt.subplot(gs[1, :2])
        plt.title('Partial Autocorrelation')
        # plt.axhline(0.2, '--')
        plt.stem(self.pacf())

        plt.subplot(gs[0, 2])
        innovations.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(innovations, plot=plt)
        plt.show()
