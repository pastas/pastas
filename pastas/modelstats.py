"""Statistics for the Pastas Model class.

Examples
--------

    >>> ml.stats.summary()
                                         Value
    Statistic
    Pearson R^2                       0.87
    Root mean squared error           0.43
    Bayesian Information Criterion    113.
    Average Deviation                 0.33
    Explained variance percentage     72.7
    Akaike InformationCriterion       25.3

Available statistics
--------------------

.. currentmodule:: pastas.modelstats.Statistics

.. autosummary::
   :nosignatures:
   :toctree: ./generated

   rmse
   rmsn
   sse
   avg_dev
   nse
   evp
   rsq
   rsq_adj
   bic
   aic
   summary
   many
   all

"""

from numpy import sqrt, log, nan
from pandas import DataFrame

from .decorators import model_tmin_tmax


class Statistics:
    # Save all statistics that can be calculated.
    ops = {'evp': 'Explained variance percentage',
           'rmse': 'Root mean squared error',
           'rmsn': 'Root mean squared noise',
           'sse': 'Sum of squares of the error',
           'avg_dev': 'Average Deviation',
           'rsq': 'Pearson R^2',
           'rsq_adj': 'Adjusted Pearson R^2',
           'bic': 'Bayesian Information Criterion',
           'aic': 'Akaike Information Criterion',
           'nse': 'Nash-Sutcliffe Efficiency'}

    def __init__(self, ml):
        """This class provides statistics to to pastas Model class.

        Parameters
        ----------
        ml: Pastas.model.Model
            ml is a time series Model that is calibrated.

        Notes
        -----
        To obtain a list of all statistics that are included type:

        >>> print(ml.stats.ops)

        """
        # Save a reference to the model.
        self.ml = ml

    def __repr__(self):
        msg = """This module contains all the statistical functions that are
included in Pastas. To obtain a list of all statistics that are included type:

    >>> print(ml.stats.ops)"""
        return msg

    # The statistical functions
    @model_tmin_tmax
    def rmse(self, tmin=None, tmax=None):
        """Root mean squared error of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        .. math:: rmse = \\sqrt{\\frac{\\sum{residuals^2}}{N}}

        where N is the number of residuals.

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        N = res.size
        return sqrt((res ** 2).sum() / N)

    @model_tmin_tmax
    def rmsn(self, tmin=None, tmax=None):
        """Root mean squared error of the noise.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Returns
        -------
        float or nan
            Return a float if noisemodel is present, nan if not.

        Notes
        -----
        .. math:: rmsn = \\sqrt{\\frac{\\sum(noise^2)}{N}}

        where N is the number of noise.

        """
        if not self.ml.settings["noise"]:
            return nan
        else:
            res = self.ml.noise(tmin=tmin, tmax=tmax).values
            N = res.size
            return sqrt((res ** 2).sum() / N)

    @model_tmin_tmax
    def sse(self, tmin=None, tmax=None):
        """Sum of the squares of the error (SSE)

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        The SSE is calculated as follows:

        .. math:: SSE = \\sum(E^2)

        Where E is an array of the residual series.

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        return (res ** 2).sum()

    @model_tmin_tmax
    def avg_dev(self, tmin=None, tmax=None):
        """Average deviation of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        .. math:: avg_dev = \\frac{\\sum(E)}{N}

        Where N is the number of the residuals.

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        return res.mean()

    @model_tmin_tmax
    def nse(self, tmin=None, tmax=None):
        """Nash-Sutcliffe coefficient for model fit .

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        Based on [nash_1970]_. (same as rsq)

        References
        ----------
        .. [nash_1970] Nash, J. E., & Sutcliffe, J. V. (1970). River flow
           forecasting through conceptual models part I-A discussion of
           principles. Journal of hydrology, 10(3), 282-290.

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        obs = self.ml.observations(tmin=tmin, tmax=tmax).values
        E = 1 - (res ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
        return E

    @model_tmin_tmax
    def evp(self, tmin=None, tmax=None):
        """Explained variance percentage.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        Commonly used statistic in time series models of groundwater levels.

        .. math:: evp = \\frac{var(h) - var(res)}{var(h)} * 100

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        obs = self.ml.observations(tmin=tmin, tmax=tmax).values
        if obs.var() == 0.0:
            return 100.
        else:
            evp = max(0.0, 100 * (1 - (res.var(ddof=0) / obs.var(ddof=0))))
        return evp

    @model_tmin_tmax
    def rsq(self, tmin=None, tmax=None):
        """Correlation between observed and simulated series.

        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax).values
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        RSS = (res ** 2.0).sum()
        TSS = ((obs - obs.mean()) ** 2.0).sum()
        return 1.0 - RSS / TSS

    @model_tmin_tmax
    def rsq_adj(self, tmin=None, tmax=None):
        """R-squared Adjusted for the number of free parameters.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        .. math:: R_{corrected} = 1-  \\frac{n-1}{n-N_{param}}*\\frac{RSS}{TSS}

        Where:

        * n = Number of observations
        * :math:`N_{param}` = Number of free parameters
        * RSS = sum of the squared residuals
        * TSS = total sum of squared residuals

        """

        obs = self.ml.observations(tmin=tmin, tmax=tmax).values
        res = self.ml.residuals(tmin=tmin, tmax=tmax).values
        N = obs.size
        RSS = (res ** 2.0).sum()
        TSS = ((obs - obs.mean()) ** 2.0).sum()
        nparam = self.ml.parameters.index.size
        return 1.0 - (N - 1.0) / (N - nparam) * RSS / TSS

    @model_tmin_tmax
    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium (BIC).

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Notes
        -----
        The Bayesian Information Criterium is calculated as follows:

        .. math:: BIC = -2 log(L) + nparam * log(N)

        Where nparam  is the number of free parameters

        Warning
        -------
        The noise is used if a noisemodel is present, otherwise the
        residuals are used.

        """
        if self.ml.settings["noise"]:
            noise = self.ml.noise(tmin=tmin, tmax=tmax).values
        else:
            noise = self.ml.residuals(tmin=tmin, tmax=tmax).values
        n = noise.size
        nparam = self.ml.parameters[self.ml.parameters.vary == True].index.size
        bic = -2.0 * log((noise ** 2.0).sum()) + nparam * log(n)
        return bic

    @model_tmin_tmax
    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium (AIC).

        Notes
        -----
        .. math:: AIC = -2 log(L) + 2 nparam

        Where

        * nparam = Number of free parameters
        * L = likelihood function for the model.

        Warning
        -------
        The noise is used if a noisemodel is present, otherwise the
        residuals are used.

        """
        if self.ml.settings["noise"]:
            noise = self.ml.noise(tmin=tmin, tmax=tmax).values
        else:
            noise = self.ml.residuals(tmin=tmin, tmax=tmax).values
        nparam = self.ml.parameters[self.ml.parameters.vary == True].index.size
        aic = -2.0 * log((noise ** 2.0).sum()) + 2.0 * nparam
        return aic

    @model_tmin_tmax
    def summary(self, tmin=None, tmax=None, stats='basic'):
        """Prints a summary table of the model statistics.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        stats : str or dict
            dictionary of the desired statistics or a string with one of the
            predefined sets. Supported options are: 'basic', 'all', and 'dutch'

        Returns
        -------
        stats : Pandas.DataFrame
            single-column DataFrame with calculated statistics

        Notes
        -----
        The set of statistics that are printed are stats by a dictionary of
        the desired statistics.

        """
        output = {
            'basic': {
                'evp': 'Explained variance percentage',
                'rmse': 'Root mean squared error',
                'avg_dev': 'Average Deviation',
                'rsq': 'Pearson R^2',
                'bic': 'Bayesian Information Criterion',
                'aic': 'Akaike Information Criterion'},
        }

        # get labels and method names for stats output
        if stats == 'all':
            # sort by key, label, method name
            selected_output = sorted([(k, l, f) for k, d in output.items()
                                      for f, l in d.items()])
        else:
            # sort by name, method name
            selected_output = sorted([(0, l, f) for f, l in
                                      output[stats].items()])

        # compute statistics
        labels_and_values = [(l, getattr(self, f)(tmin=tmin, tmax=tmax))
                             for _, l, f in selected_output]
        labels, values = zip(*labels_and_values)

        stats = DataFrame(index=list(labels), data=list(values),
                          columns=['Value'])
        stats.index.name = 'Statistic'
        return stats

    @model_tmin_tmax
    def many(self, tmin=None, tmax=None, stats=None):
        """This method returns the values for a provided list of statistics.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        stats: list, optional
            list of statistics that need to be calculated.

        Returns
        -------
        data: pandas.DataFrame

        """
        if stats is None:
            stats = ['evp', 'rmse', 'rmsn', 'rsq']

        data = DataFrame(index=[0], columns=stats)
        for k in stats:
            data.iloc[0][k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        return data

    @model_tmin_tmax
    def all(self, tmin=None, tmax=None):
        """Returns a dictionary with all the statistics.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Returns
        -------
        stats: pd.DataFrame
            Dataframe with all possible statistics

        """
        stats = DataFrame(columns=['Value'])
        for k in self.ops.keys():
            stats.loc[k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        return stats
