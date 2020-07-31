"""The following methods may be used for the diagnostic checking of the
residual time series of a calibrated (Pastas) model.

.. codeauthor:: R.A Collenteur

.. currentmodule:: pastas.stats.tests

.. autosummary::
   :nosignatures:
   :toctree: generated/

    durbin_watson
    ljung_box
    runs_test
    stoffer_toloi
    diagnostics
    plot_acf
    plot_diagnostics

"""

from logging import getLogger

import matplotlib.pyplot as plt
from numpy import sqrt, cumsum, nan, zeros, arange, finfo
from pandas import DataFrame
from scipy.stats import chi2, norm, shapiro, normaltest, probplot

from .core import acf as get_acf

logger = getLogger(__name__)
__all__ = ["durbin_watson", "ljung_box", "runs_test", "stoffer_toloi",
           "diagnostics", "plot_acf", "plot_diagnostics"]


def durbin_watson(series=None):
    """Durbin-Watson test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series, optional
        residuals series

    Returns
    -------
    dw_stat: float
        The method returns the Durbin-Watson test statistic.

    Notes
    -----
    The Durban Watson statistic ([durbin_1951]_, [Fahidy_2004]_) can be used
    to make a statement on the correlation between the values. The formula
    to calculate the Durbin-Watson statistic (DW) is:

    .. math::
        DW = 2 * (1 - \\rho)

    where acf is the autocorrelation of the series for lag s. By
    definition, the value of DW is between 0 and 4. A value of zero
    means complete negative correlation and 4 indicates complete
    positive autocorrelation. A value of zero means no autocorrelation.

    References
    ----------
    .. [durbin_1951] Durbin, J., & Watson, G. S. (1951). Testing for serial
      correlation in least squares regression. II. Biometrika, 38(1/2),
      159-177.

    .. [Fahidy_2004] Fahidy, T. Z. (2004). On the Application of Durbin-Watson
      Statistics to Time-Series-Based Regression Models. CHEMICAL ENGINEERING
      EDUCATION, 38(1), 22-25.

    TODO
    ----
    Compare calculated statistic to critical values, which are problematic
    to calculate and should probably come from a predefined table.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                data=np.random.rand(1000))
    >>>result = ps.stats.durbin_watson(res)

    """
    if not series.index.inferred_freq:
        logger.warning("Caution: The Durbin-Watson test should only be used "
                       "for time series with equidistant time steps.")

    rho = series.autocorr(lag=1)  # Take the first value of the ACF

    dw_stat = 2 * (1 - rho)
    p = nan  # NotImplementedYet
    return dw_stat, p


def ljung_box(series=None, lags=365, nparam=0, full_output=False):
    """Ljung-box test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series, optional
        series to calculate the autocorrelation for that is used in the
        Ljung-Box test.
    lags: int, optional
        The maximum lag to compute the Ljung-Box test statistic for.
    nparam: int, optional
        NUmber of calibrated parameters in the model.
    full_output: bool, optional
        Return the result of the test as a boolean (True) or not (False).

    Returns
    -------
    q_stat: float
        The computed Q test statistic.
    pval: float
        The probability of the computed Q test statistic.

    Notes
    -----
    The Ljung-Box test [Ljung_1978]_ can be used to test autocorrelation in the
    residuals series which are used during optimization of a model. The
    Ljung-Box Q-test statistic is calculated as :

    .. math::
        Q(k) = n * (n + 2) * \\sum(\\frac{\\rho^2(k)}{n - k}

    where $k$ are the lags to calculate the autocorrelation for,
    $n$ is the number of observations and :math:`\\rho(k)` is the
    autocorrelation for lag $k$. The Q-statistic can be compared to the
    value of a Chi-squared distribution to check if the Null hypothesis (no
    autocorrelation) is rejected or not. The hypothesis is rejected when:

    .. math::
        Q(k) > \\chi^2_{\\alpha, h}

    Where :math:`\\alpha` is the significance level and $h$ is the degree of
    freedom defined by $h = n - p$ where $p$ is the number of parameters
    in the model.

    References
    ----------
    .. [Ljung_1978] Ljung, G. and Box, G. (1978). On a Measure of Lack of Fit
      in Time Series Models, Biometrika, 65, 297-303.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> q_stat, pval = ps.stats.ljung_box(res)

    See Also
    --------
    pastas.stats.acf
        This method is called to compute the autocorrelation function.

    """
    if not series.index.inferred_freq:
        logger.warning("Caution: The Ljung-Box test should only be used "
                       "for time series with equidistant time steps. "
                       "Consider using ps.stats.stoffer_toloi instead.")

    acf = get_acf(series, lags=lags, bin_method="regular")
    nobs = series.index.size

    # Drop zero-lag from the acf and drop nan-values as k > 0
    acf = acf.drop(0, errors="ignore").dropna()
    lags = arange(1, len(acf) + 1)

    q_stat = nobs * (nobs + 2) * cumsum(acf.values ** 2 / (nobs - lags))
    dof = max(lags[-1] - nparam, 1)
    pval = chi2.sf(q_stat, df=dof)

    if full_output:
        result = DataFrame(data={"Q Stat": q_stat, "P-value": pval},
                           index=acf.index)
        return result
    else:
        return q_stat[-1], pval[-1]


def runs_test(series, cutoff="mean"):
    """Runs test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series
        Time series to test for autocorrelation.
    cutoff: str or float, optional
        String set to "mean" or "median" or a float to use as the cutoff.

    Returns
    -------
    z_stat: float
        Runs test statistic.
    pval: float
        p-value for the test statistic, based on a normal distribution.

    Notes
    -----
    Distribution free test to check if a time series exhibits significant
    autocorrelation [bradley_1968]_. If :math:`|Z| \\geq Z_{1-\\frac{\\alpha}{
    2}}` then the null hypothesis (Ho) is rejected.

    - Ho: The series is a result of a random process
    - Ha: The series is not the result of a random process

    References
    ----------
    .. [bradley_1968] Bradley, J. V. (1968). Distribution-free statistical
      tests.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> stat, pval = ps.stats.runs_test(res)

    """
    # Make dichotomous sequence
    r = series.values.copy()
    if cutoff == "mean":
        cutoff = r.mean()
    elif cutoff == "median":
        cutoff = r.median()
    elif isinstance(cutoff, float):
        pass
    else:
        raise NotImplementedError("Cutoff criterion {} is not "
                                  "implemented".format(cutoff))

    r[r > cutoff] = 1
    r[r < cutoff] = 0

    # Calculate number of positive and negative noise
    n_pos = r.sum()
    n_neg = r.size - n_pos

    # Calculate the number of runs
    runs = r[1:] - r[0:-1]
    n_runs = sum(abs(runs)) + 1

    # Calculate the expected number of runs and the standard deviation
    n_neg_pos = 2.0 * n_neg * n_pos
    n_runs_exp = n_neg_pos / (n_neg + n_pos) + 1
    n_runs_std = (n_neg_pos * (n_neg_pos - n_neg - n_pos)) / \
                 ((n_neg + n_pos) ** 2 * (n_neg + n_pos - 1))

    # Calculate Z-statistic and pvalue
    z_stat = (n_runs - n_runs_exp) / sqrt(n_runs_std)
    pval = 2 * norm.sf(abs(z_stat))

    return z_stat, pval


def stoffer_toloi(series, lags=365, nparam=0, freq="D"):
    """Adapted Ljung-Box test to deal with missing data [stoffer_1992]_.

    Parameters
    ----------
    series: pandas.Series
        Time series to compute the adapted Ljung-Box statistic for.
    lags: int, optional
        If lags is None, then the default maximum lag is 365. The units of
        this variable depend on the frequency chosen with the freq keyword.
    nparam: int, optional
        Number of parameters of the noisemodel.
    freq: str, optional
        String with the frequency to resample the time series to.

    Returns
    -------
    qm: float
        Adapted Ljung-Box test statistic.
    pval: float
        p-value for the test statistic, based on a chi-squared distribution.

    Notes
    -----
    stoffer-toloi test can handle missing data (nan's) in the input time
    series.

    Reference
    ---------
    .. [stoffer_1992] Stoffer, D. S., & Toloi, C. M. (1992). A note on the
       Ljung—Box—Pierce stoffer_toloi statistic with missing data. Statistics &
       probability letters, 13(5), 391-396.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                data=np.random.rand(1000))
    >>> result = ps.stats.stoffer_toloi(res)

    """
    series = series.asfreq(freq=freq)  # Make time series equidistant

    # TODO: Check if minimum frequency of the is (much) higher than freq and
    #  raise a warning. See also #https://github.com/pastas/pastas/issues/228

    nobs = series.size
    z = (series - series.mean()).fillna(0.0)
    y = z.to_numpy()
    yn = series.notna().to_numpy()

    dz0 = (y ** 2).sum() / nobs
    da0 = (yn ** 2).sum() / nobs
    de0 = dz0 / da0

    # initialize
    dz = zeros(lags)
    da = zeros(lags)
    de = zeros(lags)

    for i in range(0, lags):
        hh = y[:-i - 1] * y[i + 1:]
        dz[i] = hh.sum() / nobs
        hh = yn[:-i - 1] * yn[i + 1:]
        da[i] = hh.sum() / (nobs - i - 1)
        if abs(da[i]) > finfo(float).eps:
            de[i] = dz[i] / da[i]

    re = de / de0
    k = arange(1, lags + 1)
    # Compute the Q-statistic
    qm = nobs ** 2 * sum(da * re ** 2 / (nobs - k))

    dof = max(lags - nparam, 1)
    pval = chi2.sf(qm, df=dof)

    return qm, pval


def diagnostics(series, alpha=0.05, nparam=0, lags=365, stats=(),
                float_fmt="{0:.2f}"):
    """Methods to compute various diagnostics checks for a time series.

    Parameters
    ----------
    series: pandas.Series
        Time series to compute the diagnostics for.
    alpha: float, optional
        significance level to use for the hypothesis testing.
    nparam: int, optional
        Number of parameters of the noisemodel.
    lags: int, optional
        Maximum number of lags (in days) to compute the autocorrelation
        tests for.
    stats: list, optional
        List with the diagnostic checks to perform. Not implemented yet.
    float_fmt: str
        String to use for formatting the floats in the returned DataFrame.

    Returns
    -------
    df: Pandas.DataFrame
        DataFrame with the information for the diagnostics checks.

    Notes
    -----
    Different tests are computed depending on the regularity of the time
    step of the provided time series. series.index.inferred_freq is used to
    determined whether or not the time steps are regular.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> ps.stats.diagnostics(res)
    Out[0]:
                      Checks Statistic P-value  Reject H0
    Shapiroo       Normality      1.00    0.86      False
    D'Agostino     Normality      1.18    0.56      False
    Runs test      Autocorr.     -0.76    0.45      False
    Durbin-Watson  Autocorr.      2.02     nan      False
    Ljung-Box      Autocorr.      5.67    1.00      False

    """
    cols = ["Checks", "Statistic", "P-value"]
    df = DataFrame(index=stats, columns=cols)

    # Shapiroo-Wilk test for Normality
    stat, p = shapiro(series)
    df.loc["Shapiroo", cols] = "Normality", stat, p,

    # D'Agostino test for Normality
    stat, p = normaltest(series)
    df.loc["D'Agostino", cols] = "Normality", stat, p

    # Runs test for autocorrelation
    stat, p = runs_test(series)
    df.loc["Runs test", cols] = "Autocorr.", stat, p

    # Do different tests depending on time step
    if series.index.inferred_freq:
        # Ljung-Box test for autocorrelation
        stat, p = ljung_box(series, nparam=nparam, lags=lags)
        df.loc["Ljung-Box", cols] = "Autocorr.", stat, p

        # Durbin-Watson test for autocorrelation
        stat, p = durbin_watson(series)
        df.loc["Durbin-Watson", cols] = "Autocorr.", stat, p
    else:
        # Stoffer-Toloi for autocorrelation
        stat, p = stoffer_toloi(series, nparam=nparam, lags=lags)
        df.loc["Stoffer-Toloi", cols] = "Autocorr.", stat, p

    df["Reject H0"] = df.loc[:, "P-value"] < alpha
    df[["Statistic", "P-value"]] = \
        df[["Statistic", "P-value"]].applymap(float_fmt.format)

    return df


def plot_acf(series, alpha=0.05, lags=365, acf_options=None, smooth_conf=True,
             ax=None, figsize=(5, 2)):
    """Method to plot the autocorrelation function.

    Parameters
    ----------
    series: pandas.Series
        Residual series to plot the autocorrelation function for.
    alpha: float, optional
        Significance level to calculate the (1-alpha)-confidence intervals.
        For 95% confidence intervals, alpha should be 0.05.
    lags: int, optional
        Maximum number of lags (in days) to compute the autocorrelation for.
    acf_options: dict, optional
        Dictionary with keyword arguments passed on to pastas.stats.acf.
    smooth_conf: bool, optional
        For irregular time series the confidence interval may be
    ax: matplotlib.axes.Axes, optional
        Matplotlib Axes instance to plot the ACF on. A new Figure and Axes
        is created when no value for ax is provided.
    figsize: Tuple, optional
        2-D Tuple to determine the size of the figure created. Ignored if ax
        is also provided.

    Returns
    -------
    ax: matplotlib.axes.Axes

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> ps.stats.plot_acf(res)

    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the autocorrelation
    if acf_options is None:
        acf_options = {}
    r = get_acf(series, full_output=True, alpha=alpha, lags=lags,
                **acf_options)

    if smooth_conf:
        conf = r.stderr.rolling(10, min_periods=1).mean().values
    else:
        conf = r.stderr.values

    ax.fill_between(r.index.days, conf, -conf, alpha=0.3)
    ax.vlines(r.index.days, [0], r.loc[:, "acf"].values)

    ax.set_xlabel("Lag [Days]")
    ax.set_xlim(0, r.index.days.max())
    ax.set_ylabel('Autocorrelation [-]')
    ax.set_title("Autocorrelation plot")

    ax.grid()
    return ax


def plot_diagnostics(series, alpha=0.05, bins=50, acf_options=None,
                     figsize=(10, 6), **kwargs):
    """Plot a window that helps in diagnosing basic model assumptions.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with the residual time series to diagnose.
    alpha: float, optional
        Significance level to calculate the (1-alpha)-confidence intervals.
    bins: int optional
        Number of bins used for the histogram. 50 is default.
    acf_options: dict, optional
        Dictionary with keyword arguments passed on to pastas.stats.acf.
    figsize: tuple, optional
        Tuple with the height and width of the figure in inches.
    **kwargs: dict, optional
        Optional keyword arguments, passed on to plt.figure.

    Returns
    -------
    axes: matplotlib.axes.Axes

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.normal(0, 1, 1000))
    >>> ps.stats.plot_diagnostics(res)

    Note
    ----
    The two right-hand side plots assume that the noise or residuals follow a
    Normal distribution.

    See Also
    --------
    pastas.stats.acf
        Method that computes the autocorrelation.
    scipy.stats.probplot
        Method use to plot the probability plot.

    """
    # Create the figure and axes
    fig = plt.figure(figsize=figsize, **kwargs)
    shape = (2, 3)
    ax = plt.subplot2grid(shape, (0, 0), colspan=2, rowspan=1)
    ax1 = plt.subplot2grid(shape, (1, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(shape, (0, 2), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid(shape, (1, 2), colspan=1, rowspan=1)

    # Plot the residuals or noise series
    ax.axhline(0, c="k")
    series.plot(ax=ax)
    ax.set_ylabel(series.name)
    ax.set_xlim(series.index.min(), series.index.max())
    ax.set_title("{} (n={:.0f}, $\\mu$={:.2f})".format(series.name,
                                                       series.size,
                                                       series.mean()))
    ax.grid()

    # Plot the autocorrelation
    plot_acf(series, alpha=alpha, acf_options=acf_options, ax=ax1)

    # Plot the histogram for normality and add a 'best fit' line
    _, bins, _ = ax2.hist(series.values, bins=bins, density=True)
    y = norm.pdf(bins, series.mean(), series.std())
    ax2.plot(bins, y, 'k--')
    ax2.set_ylabel("Probability density")
    ax2.set_title("Histogram")

    # Plot the probability plot
    probplot(series, plot=ax3, dist="norm", rvalue=True)
    c = ax.get_lines()[1].get_color()
    ax3.get_lines()[0].set_color(c)
    ax3.get_lines()[1].set_color("k")

    plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    return fig.axes
