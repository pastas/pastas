"""This module contains methods for diagnosing the time series models for
its statistic assumptions.

.. codeauthor:: R.A Collenteur

.. currentmodule:: pastas.stats.tests

.. autosummary::
   :nosignatures:
   :toctree: generated/

    durbin_watson
    ljung_box
    runs_test

"""

import matplotlib.pyplot as plt
from numpy import sqrt, cumsum, nan
from pandas import DataFrame
from scipy.stats import chi2, norm, shapiro, normaltest, probplot

from .core import acf as get_acf

__all__ = ["ljung_box", "runs_test", "durbin_watson", "plot_acf",
           "plot_diagnostics"]


def durbin_watson(series=None, acf=None, alpha=0.05, **kwargs):
    """Durbin-Watson test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series, optional
        residuals series
    acf: pandas.Series, optional
        the autocorrelation function.
    alpha: float
        default alpha=0.05
    kwargs:
        all keyword arguments are passed on to the acf function.

    Returns
    -------
    dw_stat: float
        The method returns the Durbin-Watson test statistic.

    Notes
    -----
    The Durban Watson statistic [1]_ [2]_ can be used to make a statement on
    the correlation between the values. The formula to calculate the Durbin
    Watson statistic (DW) is:

    .. math::

        DW = 2 * (1 - \\rho)

    where acf is the autocorrelation of the series for lag s. By
    definition, the value of DW is between 0 and 4. A value of zero
    means complete negative correlation and 4 indicates complete
    positive autocorrelation. A value of zero means no autocorrelation.

    References
    ----------
    .. [1] Durbin, J., & Watson, G. S. (1951). Testing for serial correlation
      in least squares regression. II. Biometrika, 38(1/2), 159-177.

    .. [2] Fahidy, T. Z. (2004). On the Application of Durbin-Watson
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
    if acf is None:
        acf = get_acf(series, **kwargs)

    rho = acf.iloc[0]  # Take the first value of the ACF

    dw_stat = 2 * (1 - rho)
    p = nan  # NotImplementedYet
    return dw_stat, p


def ljung_box(series=None, acf=None, nobs=None, alpha=0.05, return_h=False,
              **kwargs):
    """Ljung-box test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series, optional
        series to calculate the autocorrelation for that is used in the
        Ljung-Box test.
    acf: pandas.Series, optional
        The autocorrelation function used in the Ljung-Box test. Using a
        pre-calculated acf will be faster. If providing the acf, nobs also
        has to be provided.
    nobs: int, optional
        Number of observations of the original time series. Has no effect
        when a series is provided.
    alpha: float, optional
        Significance level to test against. Float values between 0 and 1.
        Default is alpha=0.05.
    return_h: bool, optional
        Return the result of the test as a boolen (True) or not (False).
    kwargs:
        The keyword arguments provided to this method will be passed on the
        the ps.stats.acf method.

    Returns
    -------
    h: bool
        True if series has significant autocorrelation for any of the lags.
    Pandas.DataFrame
        DataFrame containing the Q test statistic and the p-value.

    Notes
    -----
    The Ljung-Box test [3]_ can be used to test autocorrelation in the
    residuals series which are used during optimization of a model. The
    Ljung-Box Q-test statistic is calculated as :

    .. math::

        Q(k) = n * (n + 2) * \\sum(\\frac{\\rho^2(k)}{n - k}

    where $k$ are the lags to calculate the autocorrelation for,
    $n$ is the number of observations and $\\rho(k)$ is the autocorrelation for
    lag $k$. The Q-statistic can be compared to the value of a
    Chi-squared distribution to check if the Null hypothesis (no
    autocorrelation) is rejected or not. The hypothesis is rejected when:

    .. math::

        Q(k) > \\chi^2_{\\alpha, h}

    Where $\\alpha$ is the significance level and $h$ is the degree of
    freedom defined by $h = n - p$ where $p$ is the number of parameters
    in the model.

    References
    ----------
    .. [3](Ljung, G. and Box, G. (1978). On a Measure of Lack of Fit in Time
      Series Models, Biometrika, 65, 297-303.)

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                data=np.random.rand(1000))
    >>>result = ps.stats.ljung_box(res)

    """
    if acf is None:
        acf = get_acf(series, **kwargs)
        nobs = series.index.size
    elif nobs is None:
        Warning("If providing an acf, nobs also has to be provided.")

    # Drop zero-lag from the acf and drop nan-values as k > 0
    acf = acf.drop(0, errors="ignore").dropna()

    lags = acf.index.days.to_numpy()

    nk = nobs - lags
    # nk[nk == 0] = 1
    q_stat = nobs * (nobs + 2) * cumsum(acf.values ** 2 / nk)

    # TODO decrease lags by number of parameters?
    dof = lags  # Degrees of Freedom for Chi-Squares Dist.
    pval = chi2.sf(q_stat, df=dof)

    if len(lags) == 1:

        if return_h:
            h = pval < alpha
            return q_stat, pval, h
        else:
            return q_stat, pval
    else:
        result = DataFrame(data={"Q Stat": q_stat, "P-value": pval},
                           index=lags)
        result.index.name = "Lags (Days)"
        if return_h:
            name = "Reject H0 (alpha={})".format(alpha)
            result[name] = pval < alpha
        return result


def runs_test(series, alpha=0.05, cutoff="mean", return_h=False):
    """Runs test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series
        Time series to test for autocorrelation.
    alpha: float, optional
        Significance level to use in the test.
    cutoff: str or float, optional
        String set to "mean" or "median" or a float to use as the cutoff.
    return_h: bool, optional
        Return the result of the test as a boolen (True) or not (False).

    Returns
    -------
    z_stat: float
        Runs test statistic
    pval: float
        p-value for the test statistic, based on a normal distribution.
    h: bool, optional
        Boolean that tells if the Null hypothesis is rejected (h=True)
        or if the fails to reject the Null (False) at significance
        level alpha. Only returned if return_h=True.

    Notes
    -----
    Distribution free test to check if a time series exhibits significant
    autocorrelation. If :math:`|Z| \\geq Z_{1-\\frac{\\alpha}{2}}` then the
    null hypothesis (Ho) is rejected.

    Ho: The series is a result of a random process
    Ha: The series is not the result of a random process

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

    if return_h:
        h = pval < alpha
        return z_stat, pval, h
    else:
        return z_stat, pval


def diagnostics(series, alpha=0.05, stats=(), float_fmt="{0:.2f}"):
    """Methods to compute various diagnostics checks for a time series.

    Parameters
    ----------
    series: pandas.Series
    alpha: float, optional
        significance level to use for the hypothesis testing.
    stats: list, optional
        List with the diagnostic checks to perform. Not implemented yet.
    float_fmt: str
        String to use for formatting the floats in the returned DataFrame.

    Returns
    -------
    df: Pandas.DataFrame
        DataFrame with the information for the diagnostics checks.

    """
    cols = ["Checks", "Statistic", "P-value"]
    df = DataFrame(index=stats, columns=cols)
    df.style.format("{:.2f}")

    # Shapiroo-Wilk test for Normality
    stat, p = shapiro(series)
    df.loc["Shapiroo", cols] = "Normality", stat, p,

    # D'Agostino test for Normality
    stat, p = normaltest(series)
    df.loc["D'Agostino", cols] = "Normality", stat, p

    # Runs test for autocorrelation
    stat, p = runs_test(series)
    df.loc["Runs test", cols] = "Autocorr.", stat, p

    # Durbin-Watson test for autocorrelation
    stat, p = durbin_watson(series, alpha=alpha)
    df.loc["Durbin-Watson", cols] = "Autocorr.", stat, p

    # Ljung-Box test for autocorrelation
    stat, p = ljung_box(series, alpha=alpha, lags=[365])
    df.loc["Ljung-Box", cols] = "Autocorr.", stat[0], p[0]

    df["Reject H0"] = df.loc[:, "P-value"] < alpha
    df[["Statistic", "P-value"]] = \
        df[["Statistic", "P-value"]].applymap(float_fmt.format)

    return df


def plot_acf(series, alpha=0.95, acf_options=None, ax=None, figsize=(5, 2)):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the autocorrelation
    if acf_options is None:
        acf_options = {}
    r = get_acf(series, output="full", alpha=alpha, **acf_options)
    conf = r.loc[:, "stderr"].values

    ax.fill_between(r.index.days, conf, -conf, alpha=0.3)
    ax.vlines(r.index.days, [0], r.loc[:, "acf"].values)

    ax.set_xlabel("Lag (Days)")
    ax.set_xlim(0, r.index.days.max())
    ax.set_ylabel('Autocorrelation')
    ax.grid()
    return ax


def plot_diagnostics(series, figsize=(10, 6), bins=50, acf_options=None,
                     alpha=0.05, **kwargs):
    """Plot a window that helps in diagnosing basic model assumptions.

    Parameters
    ----------
    series:
    figsize: tuple, optional
        Tuple with the height and width of the figure in inches.
    bins: int optional
        Number of bins used for the histogram. 50 is default.
    acf_options: dict, optional
        Dictionary with keyword arguments passed on to pastas.stats.acf.
    alpha: float, optional
        Significance level to calculate the (1-alpha)-confidence intervals.
    **kwargs: dict, optional
        Optional keyword arguments, passed on to plt.figure.

    Returns
    -------
    axes: list of matplotlib.axes.Axes

    Examples
    --------
    >>> axes = ml.plots.diagnostics()

    Note
    ----
    This plot assumed that the noise or residuals follow a Normal
    distribution.

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
    ax.grid()

    # Plot the autocorrelation
    plot_acf(series, alpha=alpha, acf_options=acf_options, ax=ax1)

    # Plot the histogram for normality and add a 'best fit' line
    _, bins, _ = ax2.hist(series.values, bins=bins, density=True)
    y = norm.pdf(bins, series.mean(), series.std())
    ax2.plot(bins, y, 'k--')
    ax2.set_ylabel("Probability density")

    # Plot the probability plot
    probplot(series, plot=ax3, dist="norm", rvalue=True)
    c = ax.get_lines()[1].get_color()
    ax3.get_lines()[0].set_color(c)
    ax3.get_lines()[1].set_color("k")

    plt.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

    return fig.axes
