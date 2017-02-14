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
        self.ml = ml  # Store reference to model for future use

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
 
    # The statistical functions

    def rmse(self, tmin=None, tmax=None):
        """Root mean squared error of the residuals.

        rmse = sqrt(sum(residuals**2) / N)

        """
        res = self.__getresiduals__(tmin, tmax)
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
        """Bayesian Information Criterium

        BIC = -2 log(L) + S_j log(N)
        S_j : Number of free parameters

        """
        innovations = self.__getinnovations__(tmin, tmax)
        N = innovations.size
        bic = -2.0 * np.log(sum(innovations ** 2.0)) + self.ml.nparam * np.log(
            N)
        return bic

    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium

        AIC = -2 log(L) + 2 S_j
        S_j : Number of free parameters

        """
        innovations = self.__getinnovations__(tmin, tmax)
        aic = -2.0 * np.log(sum(innovations ** 2.0)) + 2.0 * self.ml.nparam
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

   
    def __seriesbykey__(self, key, tmin=None, tmax=None):  
        """Summary
        Worker function for GHG and GLG statistcs. 
        
        Parameters
        ----------
        key : None, optional
            timeseries key ('observations' or 'simulated')
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
       
        Returns
        -------
        TYPE
            Description
        """
        if key == 'observations':
            series = self.__getobservations__(tmin=tmin, tmax=tmax)
        elif key == 'simulated':
            series = self.__getsimulated__(tmin=tmin, tmax=tmax)
        else:
            raise ValueError('no timeseries with key {key:}'.format(key=key))
        return series

    def qGHG(self, key='observations', tmin=None, tmax=None, q=0.875):
        """Summary
        Gemiddeld Hoogste Grondwaterstand (GHG)
        Approximated by taking a quantile of the timeseries values. 
        
        This Dutch groundwater statistic is also called MHGL (Mean High Groundwater Level)
        
        Parameters
        ----------
        key : None, optional
            timeseries key ('observations' or 'simulated')
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        q : float, optional
            quantile, fraction of exceedance (default 0.875)
        
        Returns
        -------
        TYPE
            Description
        """
        series = self.__seriesbykey__(key=key, tmin=tmin, tmax=tmax)
        return series.quantile(q)

    def qGLG(self, key='observations', tmin=None, tmax=None, q=0.125):
        """Summary
        Gemiddeld Laagste Grondwaterstand (GLG)
        Approximated by taking a quantile of the timeseries values. 
        
        This Dutch groundwater statistic is also called MLGL (Mean Low Groundwater Level)
                
        Parameters
        ----------
        key : None, optional
            timeseries key ('observations' or 'simulated')
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        q : float, optional
            quantile, fraction of exceedance (default 0.125)
        
        Returns
        -------
        TYPE
            Description
        """
        series = self.__seriesbykey__(key=key, tmin=tmin, tmax=tmax)
        return series.quantile(q)

    def qGVG(self, key='observations', tmin=None, tmax=None):
        """Summary
        Gemiddeld Voorjaarsgrondwaterstand (GVG)
        Approximated by taking the median of the values in the 
        period between 15 March and 15 April.
        
        This Dutch groundwater statistic is also called MSGL (Mean Spring Groundwater Level)
        
        Parameters
        ----------
        key : None, optional
            timeseries key ('observations' or 'simulated')
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        
        Returns
        -------
        TYPE
            Description
        """
        series = self.__seriesbykey__(key=key, tmin=tmin, tmax=tmax)
        isinspring = lambda x: (((x.month == 2) and (x.day >= 15)) or 
                            ((x.month == 3) and (x.day < 16)))
        inspring = series.index.map(isinspring)
        return series.loc[inspring].median()

    def dGHG(self, tmin=None, tmax=None):
        """
        Difference in GHG between simulated and observed values
        
        Parameters
        ----------
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        
        Returns
        -------
        TYPE
            Description
        """
        return (self.qGHG(key='simulated', tmin=tmin, tmax=tmax) - 
                self.qGHG(key='observations', tmin=tmin, tmax=tmax))

    def dGLG(self, tmin=None, tmax=None):
        """
        Difference in GLG between simulated and observed values
        
        Parameters
        ----------
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        
        Returns
        -------
        TYPE
            Description
        """
        return (self.qGLG(key='simulated', tmin=tmin, tmax=tmax) - 
                self.qGLG(key='observations', tmin=tmin, tmax=tmax))

    def dGVG(self, tmin=None, tmax=None):
        """
        Difference in GVG between simulated and observed values
        
        Parameters
        ----------
        tmin, tmax: Optional[pd.Timestamp]
            Time indices to use for the simulation of the time series model.
        
        Returns
        -------
        TYPE
            Description
        """
        return (self.qGVG(key='simulated', tmin=tmin, tmax=tmax) - 
                self.qGVG(key='observations', tmin=tmin, tmax=tmax))

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
        series = self.__getallseries__(tmin=tmin, tmax=tmax)
        series.describe()

    def plot_diagnostics(self, tmin=None, tmax=None):
        innovations = self.__getinnovations__(tmin=tmin, tmax=tmax)

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

    def summary(self, selected='basic', tmin=None, tmax=None):
        """Prints a summary table of the model statistics. The set of statistics
        that are printed are selected by a dictionary of the desired statistics.
        
        Parameters
        ----------
        selected_output : str or dict
            dictionary of the desired statistics or a string with one of the
            predefined sets. Supported options are: 'basic', 'all', and 'dutch'
        tmin
        
        tmax : None, optional
            Description
        tmax
        
        Returns
        -------
        stats : Pandas Dataframe
            single-column dataframe with calculated statistics        
        
        """

        output = {
                'basic': {
                    'evp': 'Explained variance percentage',
                    'rmse': 'Root mean squared error',
                    'avg_dev': 'Average Deviation',
                    'pearson': 'Pearson R^2',
                    'bic': 'Bayesian Information Criterion',
                    'aic': 'Akaike Information Criterion'},                    
                'dutch': {
                    'qGHG': 'Gemiddeld Hoge Grondwaterstand',
                    'qGLG': 'Gemiddeld Lage Grondwaterstand',
                    'qGVG': 'Gemiddelde Voorjaarsgrondwaterstand',
                    'dGHG': 'Verschil Gemiddeld Hoge Grondwaterstand',
                    'dGLG': 'Verschil Gemiddeld Lage Grondwaterstand',
                    'dGVG': 'Verschil Gemiddelde Voorjaarsgrondwaterstand'},
                    }

        if selected == 'all':
            selected_output = sorted([(k, n, f) for k, d in output.items()
                for f, n in d.items()])
        else:
            selected_output = sorted([(0, n, f) for f, n in
                output[selected].items()])

        # compute statistics
        names_and_values = [(n, getattr(self, f)(tmin=tmin, tmax=tmax))
            for _, n, f in selected_output]
        names, values = zip(*names_and_values)

        stats = pd.DataFrame(index=list(names), data=list(values),
            columns=['Value'])
        stats.index.name = 'Statistic'
        return stats
