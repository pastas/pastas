from numpy import abs, sqrt
from scipy.stats import chi2, norm

from .core import acf

__all__ = ["ljung_box", "runs_test", "durbin_watson", ]


def durbin_watson(series, tmin=None, tmax=None, **kwargs):
    """Method to calculate the durbin watson statistic.

    Parameters
    ----------
    series: pandas.Series
        the autocorrelation function.
    tmin: str
    tmax: str

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
    .. [LB] Ljung, G. and Box, G. (1978). "On a Measure of Lack of Fit in Time
    Series Models", Biometrika, 65, 297-303.

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


def runs_test(series, tmin=None, tmax=None, cutoff="mean"):
    """Runs test to test for serial autocorrelation.

    Parameters
    ----------
    series: pandas.Series
        Series to perform the runs test on.
    tmin
    tmax
    cutoff: str or float
        String set to "mean" or "median" or a float to use as the cutoff.

    Returns
    -------
    z: float
    pval: float

    """
    # Make dichotomous sequence
    R = series.copy()
    if cutoff == "mean":
        cutoff = R.mean()
    elif cutoff == "median":
        cutoff = R.median()

    R[R > cutoff] = 1
    R[R < cutoff] = 0

    # Calculate number of positive and negative noise
    n_pos = R.sum()
    n_neg = R.index.size - n_pos

    # Calculate the number of runs
    runs = R.iloc[1:].values - R.iloc[0:-1].values
    n_runs = sum(abs(runs)) + 1

    # Calculate the expected number of runs and the standard deviation
    n_neg_pos = 2.0 * n_neg * n_pos

    n_runs_exp = n_neg_pos / (n_neg + n_pos) + 1

    n_runs_std = (n_neg_pos * (n_neg_pos - n_neg - n_pos)) / \
                 ((n_neg + n_pos) ** 2 * (n_neg + n_pos - 1))

    # Calculate Z-statistic and pvalue
    z = (n_runs - n_runs_exp) / sqrt(n_runs_std)
    pval = 2 * norm.sf(abs(z))

    return z, pval
