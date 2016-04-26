import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import probplot

"""Statistics for time series models



Usage
-----

>>> ml_stats = Statistics(ml)
>>> ml_stats.summary()

Statistic:                       Value
-----------------------------  -------
Explained variance percentage  92.6574
Pearson R^2                     0.9631

TODO
----
- ACF for irregular timesteps
- PACF for irregular timesteps
- Nash-Sutcliffe
- portmanteau test (ljung-Box & Box-Pierce)
-

"""


class Statistics(object):
    """

    Methods
    -------
    rmse
    evp
    avg_dev
    pearson
    r_corrected

    """
    def __init__(self, ml):
        if ml.fit.success is not True:
            'Model optimization was not succesfull, make sure the model is solved' \
            'Properly.'
        self.h = ml.simulate()
        self.oseries = ml.oseries
        self.res = ml.oseries -self.h
        self.N = len(self.h)
        self.odelt = ml.odelt
        self.innovations = ml.noisemodel.simulate(self.res, self.odelt)
        self.N_param = ml.fit.nvarys

    def rmse(self):
        """Root mean squared error of the residuals.

        rmse = sqrt(sum(residuals**2) / N)

        """
        return np.sqrt(sum(self.res**2) / self.N)

    def avg_dev(self):
        """Average deviation of the residuals.

        """
        return np.mean(self.res)

    def evp(self):
        """Explained variance percentage.

        evp = (var(h) - var(res)) / var(h) * 100%
        """
        return (np.var(self.h) - np.var(self.res)) / np.var(self.h) * 100.0

    def pearson(self):
        """Correlation between observed and simulated series.

        """
        return np.corrcoef(self.h, self.oseries)[0, 1]

    def r_corrected(self):
        """Corrected R-squared

        R_corrected = 1 - (n-1) / (n-N_param) * RSS/TSS

        Returns
        -------

        """
        RSS = sum(self.res ** 2.0)
        TSS = sum((self.oseries - self.oseries.mean())**2.0)
        return 1.0 - (self.N -1.0) / (self.N - self.N_param) * RSS / TSS

    def bic(self):
        """Bayesian Information Criterium

        BIC = -2 log(L) + S_j log(N)
        S_j : Number of free parameters

        Returns
        -------

        """
        bic = -2.0 * np.log(sum(self.innovations**2.0)) + self.N_param * np.log(
            self.N)
        return bic

    def aic(self):
        """Akaike Information Criterium

        AIC = -2 log(L) + 2 S_j
        S_j : Number of free parameters

        Returns
        -------

        """
        aic = -2.0 * np.log(sum(self.innovations**2.0)) + 2.0 * self.N_param
        return aic

    def acf(self, nlags=20):
        """Autocorrelation function.

        TODO
        ----
        Compute autocorrelation for irregulat time steps.

        """
        return acf(self.innovations, nlags=nlags)

    def pacf(self, nlags=20):
        """Partial autocorrelation function.

        http://statsmodels.sourceforge.net/devel/_modules/statsmodels/tsa/stattools.html#acf
            
        TODO
        ----
        Compute  partial autocorrelation for irregulat time steps.
        """
        return pacf(self.innovations, nlags=nlags)

    def plot_diagnostics(self):
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
        self.innovations.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(self.innovations, plot=plt)
        plt.show()



    def summary(self, output='basic'):
        basic = {'evp' : 'Explained variance percentage', 'rmse' : 'Root mean '
                                                                   'squared error',
                 'avg_dev' : 'Average Deviation', 'pearson' : 'Pearson R^2',
                 'bic' : 'Bayesian Information Criterion', 'aic' : 'Akaike '
                                                                   'Information '
                                                                   'Criterion'}
        stats =[]
        statsvalue =[]
        header = ['Statistic:','Value']
        if output is 'basic':
            for k in basic:
                stats.append(basic[k])
                statsvalue.append(getattr(self, k)())
        print tabulate(zip(stats, statsvalue), headers=header, floatfmt=".4f")
