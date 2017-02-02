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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot

from statsmodels.tsa.stattools import acf, pacf


class Statistics(object):
    def __init__(self, ml):
        """This class is inherited by the Model class of Pastas after a
        model is calibrated.

        ml: Pastas Model
            ml is a time series Model that is calibrated.
        """
        self.ml = ml

    # The statistical functions

    def rmse(self, tmin=None, tmax=None):
        """Root mean squared error of the residuals.

        rmse = sqrt(sum(residuals**2) / N)

        """
        res = self.__getresiduals__(tmin, tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def rmsi(self, tmin=None, tmax=None):
        """Root mean squared error of the innovations.

        rmsi = sqrt(sum(innovations**2) / N)

        """
        res = self.__getinnovations__(tmin, tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def sse(self, tmin=None, tmax=None):
        res = self.__getresiduals__(tmin, tmax)
        return sum(res ** 2)

    def avg_dev(self, tmin=None, tmax=None):
        """Average deviation of the residuals.

        """
        res = self.__getresiduals__(tmin, tmax)
        return res.mean()

    def evp(self, tmin=None, tmax=None):
        """Explained variance percentage.

        evp = (var(h) - var(res)) / var(h) * 100%
        """

        res = self.__getresiduals__(tmin, tmax)
        obs = self.__getobservations__(tmin, tmax)
        return (np.var(obs) - np.var(res)) / np.var(obs) * 100.0

    def pearson(self, tmin=None, tmax=None):
        """Correlation between observed and simulated series.

        """

        sim = self.__getsimulated__(tmin, tmax)
        obs = self.__getobservations__(tmin, tmax)
        sim = sim[obs.index]  # Make sure to correlate the same in time.
        return np.corrcoef(sim, obs)[0, 1]

    def r_corrected(self, tmin=None, tmax=None):
        """Corrected R-squared

        R_corrected = 1 - (n-1) / (n-N_param) * RSS/TSS

        """

        obs = self.__getobservations__(tmin, tmax)
        res = self.__getresiduals__(tmin, tmax)
        N = obs.size

        RSS = sum(res ** 2.0)
        TSS = sum((obs - obs.mean()) ** 2.0)
        return 1.0 - (N - 1.0) / (N - self.ml.nparam) * RSS / TSS

    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium.

        BIC = -2 log(L) + nparam * log(N)
        nparam : Number of free parameters

        """
        innovations = self.__getinnovations__(tmin, tmax)
        N = innovations.size
        nparam = len(self.ml.parameters[self.ml.parameters.vary == True])
        bic = -2.0 * np.log(sum(innovations ** 2.0)) + nparam * np.log(N)
        return bic

    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium.

        AIC = -2 log(L) + 2 nparam
        nparam : Number of free parameters
        L: likelihood function for the model.

        """
        innovations = self.__getinnovations__(tmin, tmax)
        nparam = len(self.ml.parameters[self.ml.parameters.vary == True])
        aic = -2.0 * np.log(sum(innovations ** 2.0)) + 2.0 * nparam
        return aic

    def acf(self, tmin=None, tmax=None, nlags=20):
        """Autocorrelation function.

        TODO
        ----
        Compute autocorrelation for irregulat time steps.

        """
        innovations = self.__getinnovations__(tmin, tmax)
        return acf(innovations, nlags=nlags)

    def pacf(self, tmin=None, tmax=None, nlags=20):
        """Partial autocorrelation function.

        http://statsmodels.sourceforge.net/devel/_modules/statsmodels/tsa/stattools.html#acf
            
        TODO
        ----
        Compute  partial autocorrelation for irregulat time steps.
        """
        innovations = self.__getinnovations__(tmin, tmax)
        return pacf(innovations, nlags=nlags)

    def all(self, tmin=None, tmax=None, stats=None):
        """Returns a dictionary with all the statistics.

        Parameters
        ----------
        tmin
        tmax

        Returns
        -------

        """
        keys = ["evp", "rmsi", "rmse", "aic", "bic", "avg_dev", "pearson", "r_corrected", "sse"]
        stats = dict()
        for k in keys:
            stats[k] = (getattr(self, k)(tmin, tmax))

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
    #         oseries = self.__getobservations__(tmin, tmax)
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
    #         oseries = self.__getobservations__(tmin, tmax)
    #         for year in np.unique(oseries.index.year):
    #             x.append(oseries['%i' % year].sort_values(ascending=True,
    #                                                       inplace=False)[
    #                      0:3].values)
    #         return np.mean(np.array(x))

    def descriptive(self, tmin=None, tmax=None):
        series = self.__getallseries__(tmin, tmax)
        series.describe()

    def plot_diagnostics(self, tmin=None, tmax=None):
        innovations = self.__getinnovations__(tmin, tmax)

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

    def summary(self, output='basic', tmin=None, tmax=None):
        """Prints a summary table of the model statistics. The set of statistics
        that are printed are selected by a dictionary of the desired statistics.

        Parameters
        ----------
        output: str or dict
            dictionary of the desired statistics or a string with one of the
            predefined sets. Supported options are: 'basic', 'all', and 'dutch'.
        tmin

        tmax

        Returns
        -------

        """

        basic = {'evp': 'Explained variance percentage', 'rmse': 'Root mean '
                                                                 'squared error',
                 'avg_dev': 'Average Deviation', 'pearson': 'Pearson R^2',
                 'bic': 'Bayesian Information Criterion', 'aic': 'Akaike '
                                                                 'Information'
                                                                 'Criterion'}

        dutch = {'GHG': 'Gemiddeld Hoog Grondwater', 'GLG': 'Gemiddeld Laag '
                                                            'Grondwater'}

        all = {'evp': 'Explained variance percentage', 'rmse': 'Root mean '
                                                               'squared error',
               'avg_dev': 'Average Deviation', 'pearson': 'Pearson R^2',
               'bic': 'Bayesian Information Criterion', 'aic': 'Akaike '
                                                               'Information'
                                                               'Criterion'}

        if type(output) == str:
            output = eval(output)
        names = output.values()
        statsvalue = []

        for k in output:
            statsvalue.append(getattr(self, k)(tmin, tmax))

        stats = pd.DataFrame(index=names, data=statsvalue, columns=['Value'])
        stats.index.name = 'Statistic'
        print(stats)

    # Private methods that return the series for a specified tmax and tmin
    # meant for internal use.

    def __getresiduals__(self, tmin=None, tmax=None):
        series = self.ml.residuals(tmin=tmin, tmax=tmax, noise=False)
        return series

    def __getsimulated__(self, tmin=None, tmax=None):
        series = self.ml.simulate(tmin=tmin, tmax=tmax)
        return series

    def __getinnovations__(self, tmin=None, tmax=None):
        series = self.ml.residuals(tmin=tmin, tmax=tmax, noise=True)
        return series

    def __getobservations__(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = self.ml.oseries.index.min()
        if tmax is None:
            tmax = self.ml.oseries.index.max()
        tindex = self.ml.oseries[tmin: tmax].index
        series = self.ml.oseries[tindex]
        return series

    def __getallseries__(self, tmin=None, tmax=None):
        """
        Method to easily obtain all four series.

        Parameters
        ----------
        tmin
        tmax

        Returns
        -------
        series: pd.Dataframe
            returns pandas dataframe with all four time series.

        """
        series = pd.DataFrame()
        series["simulated"] = self.__getsimulated__(tmin=tmin, tmax=tmax)
        series["observations"] = self.__getobservations__(tmin=tmin, tmax=tmax)
        series["residuals"] = self.__getresiduals__(tmin=tmin, tmax=tmax)
        series["innovations"] = self.__getinnovations__(tmin=tmin, tmax=tmax)
        return series
