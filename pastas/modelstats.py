"""The following methods may be used to describe the fit between the model
simulation and the observations.

.. currentmodule:: pastas.modelstats.Statistics

.. autosummary::
   :nosignatures:
   :toctree: ./generated

   summary

Examples
========
These methods may be used as follows.


    >>> ml.stats.summary(stats=["rmse", "mae", "nse"])
                  Value
    Statistic
    rmse       0.114364
    mae        0.089956
    nse        0.929136

"""

from numpy import nan
from pandas import DataFrame

from .decorators import model_tmin_tmax, PastasDeprecationWarning
from .stats import metrics, diagnostics


class Statistics:
    # Save all statistics that can be calculated.
    ops = ["rmse", "rmsn", "sse", "mae", "nse", "evp", "rsq", "bic", "aic", ]

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

    @model_tmin_tmax
    def rmse(self, tmin=None, tmax=None, weighted=False):
        """Root mean squared error of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.rmse

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.rmse(res=res, weighted=weighted)

    @model_tmin_tmax
    def rmsn(self, tmin=None, tmax=None, weighted=False):
        """Root mean squared error of the noise.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        Returns
        -------
        float or nan
            Return a float if noisemodel is present, nan if not.

        See Also
        --------
        pastas.stats.rmse

        """
        if not self.ml.settings["noise"]:
            return nan
        else:
            res = self.ml.noise(tmin=tmin, tmax=tmax)
            return metrics.rmse(res=res, weighted=weighted)

    @model_tmin_tmax
    def sse(self, tmin=None, tmax=None):
        """Sum of the squares of the error (SSE)

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.sse

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.sse(res=res)

    @model_tmin_tmax
    def mae(self, tmin=None, tmax=None, weighted=False):
        """Mean Absolute Error (MAE) of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.mae

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.mae(res=res, weighted=weighted)

    @model_tmin_tmax
    def nse(self, tmin=None, tmax=None, weighted=False):
        """Nash-Sutcliffe coefficient for model fit .

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.nse

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.nse(obs=obs, res=res, weighted=weighted)

    @model_tmin_tmax
    def evp(self, tmin=None, tmax=None, weighted=False):
        """Explained variance percentage.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.evp

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.evp(obs=obs, res=res, weighted=weighted)

    @model_tmin_tmax
    def rsq(self, tmin=None, tmax=None):
        """R-squared.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.rsq

        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.rsq(obs=obs, res=res)

    @model_tmin_tmax
    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium (BIC).

        Parameters
        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.bic

        """
        nparam = self.ml.parameters.index.size
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.bic(res=res, nparam=nparam)

    @model_tmin_tmax
    def aic(self, tmin=None, tmax=None):
        """Akaike Information Criterium (AIC).

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.rsq

        """
        nparam = self.ml.parameters.index.size
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.aic(res=res, nparam=nparam)

    @model_tmin_tmax
    def summary(self, tmin=None, tmax=None, stats=None):
        """Returns a Pandas DataFrame with goodness-of-fit metrics.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        stats: list, optional
            list of statistics that need to be calculated. If nothing is
            provided, all statistics are returned.

        Returns
        -------
        stats : Pandas.DataFrame
            single-column DataFrame with calculated statistics

        Examples
        --------

        >>> ml.stats.summary()

        or

        >>> ml.stats.summary(stats=["mae", "rmse"])

        """
        if stats is None:
            stats_to_compute = self.ops
        else:
            stats_to_compute = stats

        stats = DataFrame(columns=['Value'])

        for k in stats_to_compute:
            stats.loc[k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        stats.index.name = 'Statistic'
        return stats

    @PastasDeprecationWarning
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

    @PastasDeprecationWarning
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
        for k in self.ops:
            stats.loc[k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        return stats

    @model_tmin_tmax
    def diagnostics(self, tmin=None, tmax=None, alpha=0.05, stats=(),
                    float_fmt="{0:.2f}"):
        if self.ml.noisemodel and self.ml.settings["noise"]:
            series = self.ml.noise(tmin=tmin, tmax=tmax)
            nparam = self.ml.noisemodel.nparam
        else:
            series = self.ml.residuals(tmin=tmin, tmax=tmax)
            nparam = 0

        return diagnostics(series=series, alpha=alpha, nparam=nparam,
                           stats=stats, float_fmt=float_fmt)
