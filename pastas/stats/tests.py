"""This module contains methods for diagnosing the time series models for
its statistic assumptions.


Autocorrelation:
    - durbin_watson
    - ljung_box
    - runs_test
    - breusch_godfrey



"""

from numpy import abs, sqrt, cumsum
from pandas import DataFrame, concat
from scipy.stats import chi2, norm

from .core import acf as get_acf

__all__ = ["ljung_box", "runs_test", "durbin_watson", ]


def durbin_watson(series=None, acf=None, alpha=0.05, **kwargs):
    """Method to calculate the durbin watson statistic.

    Parameters
    ----------
    series: pandas.Series
        the autocorrelation function.
    tmin: str
    tmax: str
    kwargs:
        all keyword arguments are passed on to the acf function.

    Returns
    -------
    DW: float

    Notes
    -----
    The Durban Watson statistic can be used to make a statement on the
    correlation between the values. The formula to calculate the Durbin
    Watson statistic (DW) is:

    .. math::

        DW = 2 * (1 - acf(s))

    where acf is the autocorrelation of the series for lag s. By
    definition, the value of DW is between 0 and 4. A value of zero
    means complete negative correlation and 4 indicates complete
    positive autocorrelation. A value of zero means no autocorrelation.

    References
    ----------
    .. [DW} Durbin, J., & Watson, G. S. (1951). Testing for serial correlation
    in least squares regression. II. Biometrika, 38(1/2), 159-177.

    .. [F] Fahidy, T. Z. (2004). On the Application of Durbin-Watson
    Statistics to Time-Series-Based Regression Models. CHEMICAL ENGINEERING
    EDUCATION, 38(1), 22-25.

    TODO
    ----
    Compare calculated statistic to critical values, which are
    problematic to calculate and should come from a predefined table.

    """
    if acf is None:
        acf = get_acf(series, **kwargs)

    DW = 2 * (1 - acf)
    DW.name = "DWtest"

    # result = DataFrame(index=lags, data={"Qtest": Qtest, "P-value": pval})
    # result.index.name = "Lags (Days)"
    # name = "Accept Ha (alpha={})".format(alpha)
    # result[name] = pval < alpha

    return DW


def ljung_box(series=None, acf=None, alpha=0.05, **kwargs):
    """Ljung-box test for autocorrelation. Either a "series" or "acf" has to be
    provided.

    Parameters
    ----------
    series: pandas.Series, optional
        series to calculate the autocorrelation for that is used in the
        Ljung-Box test.
    acf: pandas.Series, optional
        The autocorrelation function used in the Ljung-Box test. Using a
        pre-calculated acf will be faster.
    alpha: float, optional
        Significance level to test against. Float values between 0 and 1.
        Default is alpha=0.05.
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
    The Ljung-Box test can be used to test autocorrelation in the
    residuals series which are used during optimization of a model. The
    Ljung-Box Q-test statistic is calculated as :

    .. math::

        Q(k) = nobs * (n + 2) * \Sum(acf^2(k) / (n - k)

    where `k` are the lags to calculate the autocorrelation for,
    nobs is the number of observations and `acf(k)` is the autocorrelation for
    lag `k`. The Q-statististic can be compared to the value of a
    Chi-squared distribution to check if the Null hypothesis (no
    autocorrelation) is rejected or not. The hypothesis is rejected when:

    .. math::

        Q(k) > Chi^2(\alpha, h)

    Where \alpha is the significance level and `h` is the degree of
    freedom defined by `h = nobs - p` where `p` is the number of parameters
    in the model.

    References
    ----------
    .. [LB] Ljung, G. and Box, G. (1978). "On a Measure of Lack of Fit in Time
    Series Models", Biometrika, 65, 297-303.

    Examples
    --------

    v = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
              data=np.random.rand(1000))
    ps.stats.acf(v)

    """
    if acf is None:
        acf = get_acf(series, **kwargs)

    # Drop zero-lag from the acf and drop nan-values
    acf = acf.drop(0, errors="ignore").dropna()

    lags = acf.index.values
    nobs = acf.index.size
    df = (nobs - lags)
    df[df==0] = 1
    Qtest = nobs * (nobs + 2) * cumsum(acf.values ** 2 / df)

    # TODO decrease lags by number of parameters?
    pval = chi2.sf(Qtest, lags)

    result = DataFrame(index=lags, data={"LBtest": Qtest, "P-value": pval})
    result.index.name = "Lags (Days)"
    name = "Accept Ha (alpha={})".format(alpha)
    result[name] = pval < alpha
    h = result[name].any()

    return h, result


def runs_test(series, alpha=0.05, cutoff="mean"):
    """Runs test to test for autocorrelation. Returns true is there is
    significant autocorrelation according to this test.

    Parameters
    ----------
    series: pandas.Series
        Series to perform the runs test on.
    alpha: float, optional
        Significance level to use in the test.
    cutoff: str or float, optional
        String set to "mean" or "median" or a float to use as the cutoff.

    Returns
    -------
    h: bool
        Boolean that tells if the alternative hypothesis is accepted (True)
        or rejected (False) at confidence level alpha.
    Z: float
        Test-statistic
    pval: float
        p-value for the test statistic, based on a normal .

    Notes
    -----
    Ho: The series is a result of a random process
    Ha: The series is not the result of a random process

    If |Z| >= Z$_1-\alpha / 2$ then the null hypothesis (Ho) is rejected and
    the alternative hypothesis (Ha) is accepted.

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
    Z = (n_runs - n_runs_exp) / sqrt(n_runs_std)
    pval = 2 * norm.sf(abs(Z))
    h = pval < alpha
    return h, Z, pval


def acf_test(series=None, acf=None, alpha=0.05, **kwargs):
    if acf is None:
        acf = get_acf(series, **kwargs)

    h, LB = ljung_box(acf=acf, alpha=alpha)
    DW = durbin_watson(acf=acf, alpha=alpha)
    h, _, _ = runs_test(series, alpha=alpha)

    report = concat([DW, LB], axis=1)

    return h, report


def breusch_godfrey(series=None, acf=None, alpha=0.05, **kwargs):
    return NotImplementedError("Method not implemented yet.")
    # result = DataFrame()
    # h = True
    # return h, result


def lilliefors(series, alpha=0.05, **kwargs):
    """Lilliefors test to test for normality of the time series

    Parameters
    ----------
    series
    alpha
    kwargs

    Returns
    -------
    h: bool
        boolean telling whether or not to reject the alternative hypothesis
        that the data is not normally distributed.
    """
    return NotImplementedError("Method not implemented yet.")
