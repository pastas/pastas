"""The following methods may be used to describe the fit between the model simulation
and the observations.

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

from pastas.typing import Model, TimestampType

from .decorators import model_tmin_tmax
from .stats import diagnostics, metrics


class Statistics:
    # Save all statistics that can be calculated.
    ops = [
        "rmse",
        "rmsn",
        "sse",
        "mae",
        "nse",
        "evp",
        "rsq",
        "kge",
        "bic",
        "aic",
        "aicc",
    ]

    def __init__(self, ml: Model):
        """This class provides statistics to pastas Model class.

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
        msg = """This module contains all the statistical functions included in Pastas.

        To obtain a list of all statistics that are included type:

    >>> print(ml.stats.ops)"""
        return msg

    @model_tmin_tmax
    def rmse(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Root mean squared error of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.rmse
        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.rmse(res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def rmsn(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Root mean squared error of the noise.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

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
            return metrics.rmse(res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def sse(
        self, tmin: TimestampType | None = None, tmax: TimestampType | None = None
    ) -> float:
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
    def mae(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Mean Absolute Error (MAE) of the residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.mae
        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.mae(res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def nse(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Nash-Sutcliffe Efficiency for model fit .

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.nse
        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.nse(obs=obs, res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def nnse(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Normalized Nash-Sutcliffe Efficiency for model fit .

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.nnse
        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)

        return metrics.nnse(obs=obs, res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def pearsonr(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Compute the (weighted) Pearson correlation (r).

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.pearsonr
        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        return metrics.pearsonr(obs=obs, sim=sim, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def evp(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Explained variance percentage.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.evp
        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.evp(obs=obs, res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def rsq(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """R-squared.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.rsq
        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.rsq(obs=obs, res=res, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def kge(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        modified: bool = False,
        **kwargs,
    ) -> float:
        """Kling-Gupta Efficiency.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.
        modified: bool, optional
            Use the modified KGE as proposed by :cite:t:`kling_runoff_2012`.

        See Also
        --------
        pastas.stats.kge
        """
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.kge(
            obs=obs, sim=sim, weighted=weighted, modified=modified, **kwargs
        )

    @model_tmin_tmax
    def kge_2012(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        weighted: bool = False,
        **kwargs,
    ) -> float:
        """Kling-Gupta Efficiency.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        weighted: bool, optional
            If weighted is True, the variances are computed using the time step
            between observations as weights. Default is False.

        See Also
        --------
        pastas.stats.kge_2012
        """
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return metrics.kge_2012(obs=obs, sim=sim, weighted=weighted, **kwargs)

    @model_tmin_tmax
    def bic(
        self, tmin: TimestampType | None = None, tmax: TimestampType | None = None
    ) -> float:
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
        nparam = self.ml.parameters["vary"].sum()
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax) * self.ml.noise_weights(
                tmin=tmin, tmax=tmax
            )
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.bic(res=res, nparam=nparam)

    @model_tmin_tmax
    def aic(
        self, tmin: TimestampType | None = None, tmax: TimestampType | None = None
    ) -> float:
        """Akaike Information Criterium (AIC).

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.aic
        """
        nparam = self.ml.parameters["vary"].sum()
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax) * self.ml.noise_weights(
                tmin=tmin, tmax=tmax
            )
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.aic(res=res, nparam=nparam)

    @model_tmin_tmax
    def aicc(
        self, tmin: TimestampType | None = None, tmax: TimestampType | None = None
    ) -> float:
        """Akaike Information Criterium with second order bias correction (AICc).

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional

        See Also
        --------
        pastas.stats.aicc
        """
        nparam = self.ml.parameters["vary"].sum()
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax) * self.ml.noise_weights(
                tmin=tmin, tmax=tmax
            )
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return metrics.aicc(res=res, nparam=nparam)

    @model_tmin_tmax
    def summary(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        stats: list[str] | None = None,
    ) -> DataFrame:
        """Returns a Pandas DataFrame with goodness-of-fit metrics.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        stats: list, optional
            list of statistics that need to be calculated. If nothing is provided,
            all statistics are returned.

        Returns
        -------
        stats : Pandas.DataFrame
            single-column DataFrame with calculated statistics.

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

        # Remove rmsn if no noise model
        if "rmsn" in stats_to_compute and not (
            self.ml.noisemodel and self.ml.settings["noise"]
        ):
            stats_to_compute.remove("rmsn")

        stats = DataFrame(columns=["Value"])

        for k in stats_to_compute:
            stats.loc[k] = getattr(self, k)(tmin=tmin, tmax=tmax)

        stats.index.name = "Statistic"
        return stats

    @model_tmin_tmax
    def diagnostics(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        alpha: float = 0.05,
        stats: tuple = (),
        float_fmt: str = "{0:.2f}",
    ) -> DataFrame:
        """Methods to compute various diagnostics checks for the noise time series. If
        no NoiseModel is used, the diagnostics are computed on the model residuals.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
            Start date of the residual / noise time series to compute the diagnostics
            checks for.
        tmax: str or pandas.Timestamp, optional
            End date of the residuals / noise time series to compute the diagnostics
            checks for.
        alpha: float, optional
            significance level to use for the hypothesis testing.
        stats: tuple, optional
            Tuple with the diagnostic checks to perform. Not implemented yet.
        float_fmt: str
            String to use for formatting the floats in the returned DataFrame.

        Returns
        -------
        df: Pandas.DataFrame
            DataFrame with the information for the diagnostics checks. The final column
            in this DataFrame report if the Null-Hypothesis is rejected. If H0 is not
            rejected (=False) the data is in agreement with one of the properties of
            white noise (e.g., normally distributed).

        See Also
        --------
        pastas.stats.diagnostics

        """
        if self.ml.noisemodel and self.ml.settings["noise"]:
            series = self.ml.noise(tmin=tmin, tmax=tmax)
            nparam = self.ml.noisemodel.nparam
        else:
            series = self.ml.residuals(tmin=tmin, tmax=tmax)
            nparam = 0

        return diagnostics(
            series=series, alpha=alpha, nparam=nparam, stats=stats, float_fmt=float_fmt
        )
