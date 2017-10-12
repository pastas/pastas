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

Examples
--------

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

* PACF for irregular timestep
* Nash-Sutcliffe

"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.stats import chi2


class Statistics:
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
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        N = res.size
        return np.sqrt(sum(res ** 2) / N)

    def rmsi(self, tmin=None, tmax=None):
        """Root mean squared error of the innovations.

        Notes
        -----
        .. math:: rmsi = sqrt(sum(innovations**2) / N)

        where N is the number of innovations.
        """
        res = self.ml.innovations(tmin=tmin, tmax=tmax)
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
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        return sum(res ** 2)

    def avg_dev(self, tmin=None, tmax=None):
        """Average deviation of the residuals.

        Notes
        -----
        .. math:: avg_dev = sum(E) / N

        Where N is the number of the residuals.

        """
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
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
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        evp = max(0.0, (np.var(obs) - np.var(res)) / np.var(obs) * 100.0)
        return evp

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
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
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

        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        N = obs.size

        RSS = sum(res ** 2.0)
        TSS = sum((obs - obs.mean()) ** 2.0)
        nparam = self.ml.parameters.index.size
        return 1.0 - (N - 1.0) / (N - nparam) * RSS / TSS

    def bic(self, tmin=None, tmax=None):
        """Bayesian Information Criterium.

        Notes
        -----
        The Bayesian Information Criterium is calculated as follows:

        .. math:: BIC = -2 log(L) + nparam * log(N)

        Where:
            nparam : Number of free parameters
        """
        innovations = self.ml.innovations(tmin=tmin, tmax=tmax)
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
        innovations = self.ml.innovations(tmin=tmin, tmax=tmax)
        nparam = len(self.ml.parameters[self.ml.parameters.vary == True])
        aic = -2.0 * np.log(sum(innovations ** 2.0)) + 2.0 * nparam
        return aic

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
                'rsq': 'Pearson R^2',
                'bic': 'Bayesian Information Criterion',
                'aic': 'Akaike Information Criterion'},
        }

        # get labels and method names for selected output
        if selected == 'all':
            # sort by key, label, method name
            selected_output = sorted([(k, l, f) for k, d in output.items()
                                      for f, l in d.items()])
        else:
            # sort by name, method name
            selected_output = sorted([(0, l, f) for f, l in
                                      output[selected].items()])

        # compute statistics
        labels_and_values = [(l, getattr(self, f)(tmin=tmin, tmax=tmax))
                             for _, l, f in selected_output]
        labels, values = zip(*labels_and_values)

        stats = pd.DataFrame(index=list(labels), data=list(values),
                             columns=['Value'])
        stats.index.name = 'Statistic'
        return stats

    def many(self, tmin=None, tmax=None, stats=None):
        """This method returns the values for a provided list of statistics.

        Parameters
        ----------
        tmin
        tmax
        stats: list
            list of statistics that need to be calculated.

        Returns
        -------

        """
        if not stats:
            stats = ['evp', 'rmse', 'rmsi', 'durbin_watson']

        data = pd.DataFrame(index=[0], columns=stats)
        for k in stats:
            data.iloc[0][k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        return data

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
            stats.loc[k] = (getattr(self, k)(tmin=tmin, tmax=tmax))

        return stats


def acf(x, lags=None, bin_width=None,
        bin_method='rectangle', tmin=None, tmax=None):
    """Method to calculate the autocorrelation for irregular timesteps.

    See Also
    --------
    stats.ccf

    """
    C = ccf(x=x, y=x, lags=lags, bin_width=bin_width,
            bin_method=bin_method, tmin=tmin, tmax=tmax)

    return C


def ccf(x, y, lags=None, bin_width=None,
        bin_method='rectangle', tmin=None, tmax=None):
    """Method to calculate the autocorrelation for irregular timesteps
    based on the slotting technique. Different methods (kernels) to bin
    the data are available.

    Parameters
    ----------
    x: pandas.Series
    y: pandas.Series
    lags: numpy.array
        numpy array containing the lags in DAYS for which the
        cross-correlation if calculated.
    bin_width
    bin_method: str
        method to determine the type of bin. Optiona are gaussian, sinc and rectangle.

    Returns
    -------
    acf: pandas.Series
        autocorrelation function.

    References
    ----------
    Rehfeld, K. & Marwan, N. & Heitzig, J. & Kurths, J. (2011). Comparison of correlation analysis techniques for irregularly sampled time series. Nonlinear Processes in Geophysics. 18. 389–404. 10.5194/npg-18-389-2011.

    """
    # Normalize the time values
    dt_x = x.index.to_series().diff() / pd.Timedelta(1, "D")
    dt_x[0] = 0.0
    t_x = (dt_x.cumsum() / dt_x.mean()).values

    dt_y = y.index.to_series().diff() / pd.Timedelta(1, "D")
    dt_y[0] = 0.0
    t_y = (dt_y.cumsum() / dt_y.mean()).values

    dt_mu = max(dt_x.mean(), dt_y.mean())

    # Create matrix with time differences
    t1, t2 = np.meshgrid(t_x, t_y)
    t = t1 - t2

    # Normalize the values
    x = (x.values - x.mean()) / x.std()
    y = (y.values - y.mean()) / y.std()

    # Create matrix for covariances
    xx, yy = np.meshgrid(x, y)
    xy = xx * yy

    if lags is None:
        lags = [0, 1, 14, 28, 180, 365]  # Default lags in Days

    # Remove lags that cannot be determined because lag < dt_min
    dt_min = min(dt_x.iloc[1:].min(), dt_y.iloc[1:].min())
    lags = [lag for lag in lags if lag > dt_min or lag is 0]

    lags = np.array(lags) / dt_mu

    # Select appropriate bin_width, default depend on bin_method
    if bin_width is None:
        # Select one of the standard options.
        bin_width = {"rectangle": 2, "sinc": 1, "gaussian": 4}
        h = 1 / bin_width[bin_method]
    else:
        h = bin_width / dt_mu

    C = np.zeros_like(lags)

    for i, k in enumerate(lags):
        # Construct the kernel for the lag
        d = np.abs(np.abs(t) - k)
        if bin_method == "rectangle":
            b = (d <= h) * 1.
        elif bin_method == "gaussian":
            b = np.exp(-d ** 2 / (2 * h ** 2)) / np.sqrt(2 * np.pi * h)
        elif bin_method == "sinc":
            NotImplementedError()
            # b = np.sin(np.pi * h * d) / (np.pi * h * d) / dt.size
        else:
            NotImplementedError(
                "bin_method %s is not implemented." % bin_method)
        c = xy * b
        C[i] = c.sum() / b.sum()
    C = C / np.abs(C).max()

    C = pd.Series(data=C, index=lags * dt_mu)

    return C


def durbin_watson(series, tmin=None, tmax=None, **kwargs):
    """Method to calculate the durbin watson statistic.

    Parameters
    ----------
    series: pandas.Series
    tmin: str
    tmax: str
    kwargs: dict
        Any keyword arguments that are passed to acf method.

    Returns
    -------
    DW: float

    Notes
    -----
    The Durban Watson statistic can be used to make a statement on the
    correlation between the values. The formula to calculate the Durbin
    Watson statistic (DW) is:

    .. math::

        DW = 2 * (1 - r(s))

    where r is the autocorrelation of the series for lag s. By
    definition, the value of DW is between 0 and 4. A value of zero
    means complete negative correlation and 4 indicates complete
    positive autocorrelation. A value of zero means no autocorrelation.

    References
    ----------
    Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression. II. Biometrika, 38(1/2), 159-177.

    Fahidy, T. Z. (2004). On the Application of Durbin-Watson Statistics to Time-Series-Based Regression Models. CHEMICAL ENGINEERING EDUCATION, 38(1), 22-25.

    TODO
    ----
    Compare calculated statistic to critical values, which are
    problematic to calculate and should come from a predefined table.

    """

    r = acf(series, tmin=tmin, tmax=tmax, **kwargs)

    DW = 2 * (1 - r)

    return DW


def ljung_box(series, tmin=None, tmax=None, n_params=5, alpha=None, **kwargs):
    """Method to calculate the ljung-box statistic

    Parameters
    ----------
    series: pandas.Series
    tmin
    tmax
    n_params: int
        Integer for the number of free model parameters.
    alpha: float
        Float values between 0 and 1.

    Returns
    -------
    Q: float
    Qtest: tuple

    Notes
    -----
    The Ljung-Box test can be used to test autocorrelation in the
    residuals series which are used during optimization of a model. The
    Ljung-Box Q-test statistic is calculated as :

    .. math::

        Q(k) = N * (n + 2) * \Sum(r^2(k) / (n - k)

    where `k` are the lags to calculate the autocorrelation for,
    N is the number of observations and `r(k)` is the autocorrelation for
    lag `k`. The Q-statististic can be compared to the value of a
    Chi-squared distribution to check if the Null hypothesis (no
    autocorrelation) is rejected or not. The hypothesis is rejected when:

    .. math::

        Q(k) > Chi^2(\alpha, h)

    Where \alpha is the significance level and `h` is the degree of
    freedom defined by `h = N - p` where `p` is the number of parameters
    in the model.

    References
    ----------
    Ljung, G. and Box, G. (1978). "On a Measure of Lack of Fit in Time Series Models", Biometrika, 65, 297-303.

    """
    r = acf(series, tmin=tmin, tmax=tmax, **kwargs)
    r = r.drop(0)  # Drop zero-lag from the acf

    N = series.index.size
    Q = N * (N + 2) * sum(r.values ** 2 / (N - r.index))

    if alpha is None:
        alpha = [0.90, 0.95, 0.99]

    h = N - n_params

    Qtest = chi2.ppf(alpha, h)

    return Q, Qtest


# Some Dutch statistics

def q_ghg(series, tmin=None, tmax=None, q=0.94):
    """Gemiddeld Hoogste Grondwaterstand (GHG) also called MHGL (Mean High
    Groundwater Level) Approximated by taking a quantile of the
    timeseries values, after resampling to daily values.

    This function does not care about series length!

    Parameters
    ----------
    key : None, optional
        timeseries key ('observations' or 'simulated')
    tmin/tmax: pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    q : float, optional
        quantile fraction of exceedance (default 0.94)

    """
    series = series.resample('d').median()
    return series.quantile(q)


def q_glg(series, tmin=None, tmax=None, q=0.06):
    """Gemiddeld Laagste Grondwaterstand (GLG) also called MLGL (Mean Low
    Groundwater Level) approximated by taking a quantile of the
    timeseries values, after resampling to daily values.

    This function does not care about series length!

    Parameters
    ----------
    key : None, optional
        timeseries key ('observations' or 'simulated')
    tmin/tmax : pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    q : float, optional
        quantile, fraction of exceedance (default 0.06)

    """
    series = series.resample('d').median()
    return series.quantile(q)


def q_gvg(series, tmin=None, tmax=None):
    """Gemiddeld Voorjaarsgrondwaterstand (GVG) also called MSGL (Mean
    Spring Groundwater Level) approximated by taking the median of the
    values in the period between 14 March and 15 April (after resampling to
    daily values).

    This function does not care about series length!

    Parameters
    ----------
    key : None, optional
        timeseries key ('observations' or 'simulated')
    tmin/tmax: pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.

    """
    series = series.resample('d').median()
    inspring = _in_spring(series)
    if np.any(inspring):
        return series.loc[inspring].median()
    else:
        return np.nan


def gxg(series, year_agg, tmin, tmax, fill_method, limit, output):
    """Worker method for classic GXG statistics. Resampling the series to
    every 14th and 28th of the month. Taking the mean of aggregated
    values per year.

    Parameters
    ----------
    year_agg : function series -> scalar
        Aggregator function to one value per year
    tmin/tmax : pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    key : str, optional
        timeseries key ('observations' or 'simulated')
    fill_method : str
        see notes below
    limit : int or None, optional
        Maximum number of timesteps to fill using fill method, use None to
        fill all.
    output : str
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values

    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values

    Raises
    ------
    ValueError
        When output argument is unknown

    Notes
    -----
    fill method for interpolation to 14th and 28th of the month see:
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.ffill.html
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.bfill.html
        * http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.interpolate.html
        * Use None to omit filling and drop NaNs

    """
    series = series.resample('d').mean()
    if fill_method is None:
        series = series.dropna()
    elif fill_method == 'ffill':
        series = series.ffill(limit=limit)
    elif fill_method == 'bfill':
        series = series.bfill(limit=limit)
    else:
        series = series.interpolate(method=fill_method, limit=limit)

    the14or28 = lambda x: (x.day == 14) or (x.day == 28)
    is14or28 = pd.Series(series.index.map(the14or28), index=series.index)
    if not np.any(is14or28):
        return np.nan
    series = series.loc[is14or28]
    yearly = series.resample('a').apply(year_agg)
    if output == 'yearly':
        return yearly
    elif output == 'mean':
        return yearly.mean()
    else:
        ValueError('{output:} is not a valid output option'.format(
            output=output))


def ghg(series, tmin=None, tmax=None, fill_method='linear', limit=15,
        output='mean'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the mean of three highest values per
    year. This function does not care about series length!

    Parameters
    ----------
    tmin/tmax : pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    key : str, optional
        timeseries key ('observations' or 'simulated')
    fill_method : str
        see .. :mod: pastas.stats.gxg
    limit : int or None, optional
        Maximum number of timesteps to fill using fill method, use None to
        fill all.
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for mean
        of yearly values.

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values

    """
    mean_high = lambda s: s.nlargest(3).mean()
    return gxg(series, mean_high, tmin=tmin, tmax=tmax,
               fill_method=fill_method, limit=limit, output=output)


def glg(series, tmin=None, tmax=None, fill_method='linear', limit=15,
        output='mean'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the mean of three lowest values per year.
    This function does not care about series length!

    Parameters
    ----------
    tmin/tmax : pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    key : str, optional
        timeseries key ('observations' or 'simulated')
    fill_method : str, optional
        see .. :mod: pastas.stats.gxg
    limit : int or None, optional
        Maximum number of timesteps to fill using fill method, use None to
        fill all.
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values

    Returns
    -------
    pd.Series or scalar
        Series of yearly values or mean of yearly values

    """
    mean_low = lambda s: s.nsmallest(3).mean()
    return gxg(series, mean_low, tmin=tmin, tmax=tmax,
               fill_method=fill_method, limit=limit, output=output)


def gvg(series, tmin=None, tmax=None, fill_method='linear', limit=15,
        output='mean'):
    """Classic method resampling the series to every 14th and 28th of
    the month. Taking the mean of the values on March 14, March 28 and
    April 14. This function does not care about series length!

    Parameters
    ----------
    tmin/tmax : pandas.Timestamp, optional
        Time indices to use for the simulation of the time series model.
    key : str, optional
        timeseries key ('observations' or 'simulated')
    fill_method : str, optional
        see .. :mod: pastas.stats.gxg
    limit : int or None, optional
        Maximum number of timesteps to fill using fill method, use None to
        fill all.
    output : str, optional
        output type 'yearly' for series of yearly values, 'mean' for
        mean of yearly values

    Returns
    -------
    pandas.Series or scalar
        Series of yearly values or mean of yearly values

    """
    return gxg(series, _mean_spring, tmin=tmin, tmax=tmax,
               fill_method=fill_method, limit=limit, output=output)


# Helper functions

def _mean_spring(series):
    """Determine mean of timeseries values in spring.

    Year aggregator function for gvg method.

    Parameters
    ----------
    series : pandas.Series
        series with datetime index

    Returns
    -------
    float
        Mean of series, or NaN if no values in spring

    """
    inspring = _in_spring(series)
    if np.any(inspring):
        return series.loc[inspring].mean()
    else:
        return np.nan


def _in_spring(series):
    """Test if timeseries index is between 14 March and 15 April.

    Parameters
    ----------
    series : pd.Series
        series with datetime index

    Returns
    -------
    pd.Series
        Boolean series with datetimeindex
    """
    isinspring = lambda x: (((x.month == 3) and (x.day >= 14)) or
                            ((x.month == 4) and (x.day < 15)))
    return pd.Series(series.index.map(isinspring), index=series.index)
