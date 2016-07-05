"""Statistics for time series models.

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
import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import probplot
from Tkinter import W, N, E, S, Tk
from ttk import Button, Label, Frame, Labelframe, Treeview, OptionMenu, Entry, \
    Notebook
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class Statistics(object):
    """
    This class contains all the statistical methods available.

    Methods
    -------
    rmse
    evp
    avg_dev
    pearson
    r_corrected

    """

    def __init__(self, ml):
        #if ml.fit.success is not True:
        #    'Model optimization was not succesfull, make sure the model is solved' \
        #    'Properly.'

        # Store all the series for quicker computation of the statistics
        self.sseries = ml.simulate()  # Simulated series
        self.oseries = ml.oseries  # Observed series
        self.rseries = ml.residuals()  # Residuals series
        self.odelt = ml.odelt  # Timestep between observations
        if ml.noisemodel: # Calculate the innovations
            self.iseries = ml.noisemodel.simulate(self.rseries, self.odelt)

        # Sture some other parameters
        self.N = len(self.sseries)  # Number of observations
        self.N_param = ml.fit.nvarys  # Numberof varying parameters
        self.ml = ml  # Store reference to model for future use

    # Return the series for a specified tmax and tmin
    def get_series(self, series, tmin=None, tmax=None):
        assert isinstance(series, pd.Series), 'Expected a Pandas Series object, ' \
                                              'got %s' % type(series)
        if tmin is None:
            tmin = self.series.index.min()
        if tmax is None:
            tmax = self.series.index.max()
        tindex = self.series[tmin: tmax].index
        series = (self.series)[tindex]
        return series

    def get_rseries(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index
        rseries = (self.rseries)[tindex]
        return rseries

    def get_sseries(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index
        sseries = self.sseries[tindex]  # Simulated series
        return sseries

    def get_iseries(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index
        iseries = self.iseries[tindex]
        return iseries

    def get_oseries(self, tmin=None, tmax=None):
        if tmin is None:
            tmin = self.oseries.index.min()
        if tmax is None:
            tmax = self.oseries.index.max()
        tindex = self.oseries[tmin: tmax].index
        oseries = self.oseries[tindex]
        return oseries

    # The statistical functions

    def rmse(self, tmin=None, tmax=None):
        """Root mean squared error of the residuals.

        rmse = sqrt(sum(residuals**2) / N)

        """
        res = self.get_rseries(tmin, tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def avg_dev(self, tmin=None, tmax=None):
        """Average deviation of the residuals.

        """
        res = self.get_rseries(tmin, tmax)
        return res.mean()

    def evp(self, tmin=None, tmax=None):
        """Explained variance percentage.

        evp = (var(h) - var(res)) / var(h) * 100%
        """

        res = self.get_rseries(tmin, tmax)
        sseries = self.get_sseries(tmin, tmax)
        return (np.var(sseries) - np.var(res)) / np.var(sseries) * 100.0

    def pearson(self, tmin=None, tmax=None):
        """Correlation between observed and simulated series.

        """

        sseries = self.get_sseries(tmin, tmax)
        oseries = self.get_oseries(tmin, tmax)
        return np.corrcoef(sseries, oseries)[0, 1]

    def r_corrected(self, tmin=None, tmax=None):
        """Corrected R-squared

        R_corrected = 1 - (n-1) / (n-N_param) * RSS/TSS

        """

        oseries = self.get_oseries(tmin, tmax)
        res = self.get_rseries(tmin, tmax)
        N = oseries.size

        RSS = sum(res ** 2.0)
        TSS = sum((oseries - oseries.mean()) ** 2.0)
        return 1.0 - (N - 1.0) / (N - self.N_param) * RSS / TSS

    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium

        BIC = -2 log(L) + S_j log(N)
        S_j : Number of free parameters

        """
        iseries = self.get_iseries(tmin, tmax)
        N = iseries.size
        bic = -2.0 * np.log(sum(iseries ** 2.0)) + self.N_param * np.log(N)
        return bic

    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium

        AIC = -2 log(L) + 2 S_j
        S_j : Number of free parameters

        """
        iseries = self.get_iseries(tmin, tmax)
        aic = -2.0 * np.log(sum(iseries ** 2.0)) + 2.0 * self.N_param
        return aic

    def acf(self, tmin=None, tmax=None, nlags=20):
        """Autocorrelation function.

        TODO
        ----
        Compute autocorrelation for irregulat time steps.

        """
        iseries = self.get_iseries(tmin, tmax)
        return acf(iseries, nlags=nlags)

    def pacf(self, tmin=None, tmax=None, nlags=20):
        """Partial autocorrelation function.

        http://statsmodels.sourceforge.net/devel/_modules/statsmodels/tsa/stattools.html#acf
            
        TODO
        ----
        Compute  partial autocorrelation for irregulat time steps.
        """
        iseries = self.get_iseries(tmin, tmax)
        return pacf(iseries, nlags=nlags)

    def GHG(self, tmin=None, tmax=None, series='oseries'):
        """GHG: Gemiddeld Hoog Grondwater (in Dutch)

        3 maximum groundwater level observations for each year divided by 3 times
        the number of years.

        Parameters
        ----------
        series: Optional[str]
            string for the series to calculate the statistic for. Supported
            options are: 'oseries'.

        """
        if series == 'oseries':
            x = []
            oseries = self.get_oseries(tmin, tmax)
            for year in np.unique(oseries.index.year):
                x.append(oseries['%i' % year].sort_values(ascending=False,
                                                          inplace=False)[
                         0:3].values)
            return np.mean(np.array(x))

    def GLG(self, tmin=None, tmax=None, series='oseries'):
        """GLG: Gemiddeld Laag Grondwater (in Dutch)

        3 minimum groundwater level observations for each year divided by 3 times
        the number of years.

        Parameters
        ----------
        series: Optional[str]
            string for the series to calculate the statistic for. Supported
            options are: 'oseries'.

        """
        if series == 'oseries':
            x = []
            oseries = self.get_oseries(tmin, tmax)
            for year in np.unique(oseries.index.year):
                x.append(oseries['%i' % year].sort_values(ascending=True,
                                                          inplace=False)[
                         0:3].values)
            return np.mean(np.array(x))

    def descriptive(self, series, tmin, tmax):
        series = self.get_series(series, tmin, tmax)
        series.describe()

    def plot_diagnostics(self, tmin=None, tmax=None):
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
        self.iseries.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(self.iseries, plot=plt)
        plt.show()

    def plot_statistics(self):
        """Plot the statistics with tabs. Experimental!!

        Returns
        -------

        """
        self.root = Tk()
        self.root.title('Model Summary')
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))

        top = Frame(self.root)
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        top.grid()

        n = Notebook(top)
        n.grid(row=0, column=0, sticky=W + E + N + S)

        f1 = Frame(n)  # first page, which would get widgets gridded into it
        f = Figure(facecolor="white", figsize=(2, 1))
        self.ts_ax = f.add_subplot(311)
        f.add_subplot(312)
        f.add_subplot(313)

        # self.ts_ax.set_position([0.05, 0.1, 0.9, 0.85]) # This should not be
        #  hardcoded
        self.ts_canvas = FigureCanvasTkAgg(f, master=f1)
        self.ts_canvas.get_tk_widget().grid(row=1, column=1, columnspan=2,
                                            rowspan=4,
                                            sticky=W + E + N + S)
        f2 = Frame(n)  # second page

        n.add(f1, text='One')
        Button(f1, text='Test').grid()

        n.add(f2, text='Two')
        Button(f2, text='Test').grid()

        self.root.mainloop()

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
                                                               'Criterion',
               'GHG': 'Gemiddeld Hoog Grondwater', 'GLG': 'Gemiddeld Laag '
                                                          'Grondwater'}

        if type(output) == str:
            output = eval(output)
        names = output.values()
        statsvalue = []

        for k in output:
            statsvalue.append(getattr(self, k)(tmin, tmax))

        stats = pd.DataFrame(index=names, data=statsvalue, columns=['Value'])
        stats.index.name = 'Statistic'
        print stats
