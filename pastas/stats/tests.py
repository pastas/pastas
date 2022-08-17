"""The following methods may be used for the diagnostic checking of the
residual time series of a calibrated (Pastas) model.

.. codeauthor:: R.A Collenteur
"""

from logging import getLogger

from numpy import arange, cumsum, finfo, median, nan, sqrt, zeros
from pandas import DataFrame, date_range, infer_freq
from pastas.stats.core import acf as get_acf
from pastas.utils import _get_time_offset, get_equidistant_series
from scipy.stats import chi2, norm, normaltest, shapiro

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
    The Durban Watson statistic ([durbin_1951]_, [Fahidy_2004]_) tests the
    null-hypothesis that the correlation between the noise values at lag one
    equals zero. The formula to calculate the Durbin-Watson statistic (DW) is:

    .. math::
        DW = \\frac{\\sum_{t=2}^{n}(\\upsilon_t-\\upsilon_{t-1}^2)}
        {\\sum_{t=1}^{n}\\upsilon_t^2}

    where $n$ is the number of values in the noise series. The test-statistic
    has a range :math:`0 \\geq DW \\leq 4`, where values of $DW < 2$ indicate a
    positive correlation and values of $DW > 2$ indicates negative
    autocorrelation. The Durbin-Watson test requires a constant time interval
    of the noise series and tests for autocorrelation at a lag of 1 time step.

    **Considerations for this test:**

    - The time series should have equidistant time steps.
    - The Durbin-Watson test tests for autocorrelation at lag 1 but not for
      larger time lags.
    - The test statistic for this test is difficult to compute and is usually
      obtained from pre-calculated tables.

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
    >>> data = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                data=np.random.rand(1000))
    >>> result = ps.stats.durbin_watson(data)
    """
    if not infer_freq(series.index):
        logger.warning("Caution: The Durbin-Watson test should only be used "
                       "for time series with equidistant time steps.")

    rho = series.autocorr(lag=1)  # Take the first value of the ACF

    dw_stat = 2 * (1 - rho)
    p = nan  # NotImplementedYet
    return dw_stat, p


def ljung_box(series=None, lags=15, nparam=0, full_output=False):
    """Ljung-box test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series, optional
        series to calculate the autocorrelation for that is used in the
        Ljung-Box test.
    lags: int, optional
        The maximum lag to compute the Ljung-Box test statistic for.
    nparam: int, optional
        Number of calibrated parameters in the model.
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
    The Ljung-Box test [Ljung_1978]_ tests the null-hypothesis that a time
    series are independently distributed up to a desired time lag $k$ and is
    computed as follows:

    .. math::
        Q(k) = n (n + 2) \\sum_{k=1}^{h} \\frac{\\rho^2(k)}{n - k}

    where :math:`\\rho_k` is the autocorrelation at lag $k$, $h$ is the
    maximum lag used for calculation, and $n$ is the number of values in the
    noise series. The computed $Q$-statistic is then compared to a critical
    value computed from a :math:`\\chi^2_{\\alpha, h-p}` distribution with a
    significance level :math:`\\alpha` and $h-p$ degrees of freedom, where $h$
    is the number of lags and $p$ the number of the noise model parameters.

    **Considerations for this test:**

    - The time series should have equidistant time steps. An adapted version
      of the Ljung-Box test is available through ps.stats.stoffer_toloi.
    - A potential problem of the Ljung-Box test is the low power of the test
      when testing for a large number of lags using a small sample size $n$.
      It has been suggested that suggested that :math:`k \\leq n/4` but also
      as low as :math:`k \\leq n/20`. If we are using daily groundwater levels
      observations, and we want to test for autocorrelation for lags up to one
      year (365 days) this means that we need between 4 and ten years of data.

    References
    ----------
    .. [Ljung_1978] Ljung, G. and Box, G. (1978). On a Measure of Lack of Fit
      in Time Series Models, Biometrika, 65, 297-303.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> stat, p = ps.stats.ljung_box(res, lags=15)
    >>> if p > alpha:
    >>>    print("Failed to reject the Null-hypothesis, no significant"
    >>>          "autocorrelation. p =", p.round(2))
    >>> else:
    >>>    print("Reject the Null-hypothesis. p =", p.round(2))

    See Also
    --------
    pastas.stats.acf
        This method is called to compute the autocorrelation function.
    pastas.stats.stoffer_toloi
        Similar method but adapted for time series with missing data.
    """
    if not infer_freq(series.index):
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


def runs_test(series, cutoff="median"):
    """Runs test for autocorrelation.

    Parameters
    ----------
    series: pandas.Series
        Time series to test for autocorrelation.
    cutoff: str or float, optional
        String set to "mean", "median", or a float value to use as the cutoff.

    Returns
    -------
    z_stat: float
        Runs test statistic.
    pval: float
        p-value for the test statistic, based on a normal distribution.

    Notes
    -----
    Wald and Wolfowitz developed [wald_1943]_ developed a distribution free
    test (i.e., no normal distribution is assumed) to test for
    autocorrelation. This test is also appropriate for non-equidistant
    time steps in the residuals time series. The Null-hypothesis is that the
    residual time series is a random sequence of positive and negative values.
    The alternative hypothesis is that they are non-random. The test statistic
    is computed as follows:

    .. math::
        Z = \\frac{R-\\bar{R}}{\\sigma_R}

    where $R$ is the number of runs, :math:`\\bar{R}` the expected number of
    runs and :math:`\\sigma_R` the standard deviation of the number of runs.
    A run is defined as the number of sequences of exclusively postitive and
    negative values in the time series.

    **Considerations for this test:**

    - Test is also applicable to time series with non-equidistant time steps.

    References
    ----------
    .. [wald_1943] Wald, A., & Wolfowitz, J. (1943). An exact test for
       randomness in the non-parametric case based on serial correlation.
       The Annals of Mathematical Statistics, 14(4), 378-388.

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> stat, pval = ps.stats.runs_test(res)
    >>> if p > alpha:
    >>>     print("Failed to reject the Null-hypothesis, no significant"
    >>>           "autocorrelation. p =", p.round(2))
    >>> else:
    >>>     print("Reject the Null-hypothesis")
    """
    # Make dichotomous sequence
    r = series.copy().to_numpy()
    if cutoff == "mean":
        cutoff = r.mean()
    elif cutoff == "median":
        cutoff = median(r)
    elif isinstance(cutoff, float):
        pass
    else:
        raise NotImplementedError(f"Cutoff criterion {cutoff} is not "
                                  f"implemented.")

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


def stoffer_toloi(series, lags=15, nparam=0, freq="D",
                  snap_to_equidistant_timestamps=False):
    """Adapted Ljung-Box test to deal with missing data [stoffer_1992]_.

    Parameters
    ----------
    series: pandas.Series
        Time series to compute the adapted Ljung-Box statistic for.
    lags: int, optional
        the number of lags to compute the statistic for. Only lags for which
        a correlation is computed are used.
    nparam: int, optional
        Number of parameters of the noisemodel.
    freq: str, optional
        String with the frequency to resample the time series to.
    snap_to_equidistant_timestamps : bool, optional
        if False (default), a sample is taken from series with equidistant 
        timesteps using pandas' reindex. Only values are kept that lie on
        those equidistant timestamps. If True, an equidistant timeseries is
        created taking as many values as possible from the original series
        which are then snapped to the nearest equidistant timestamp.

    Returns
    -------
    qm: float
        Adapted Ljung-Box test statistic.
    pval: float
        p-value for the test statistic, based on a chi-squared distribution.

    Notes
    -----
    Stoffer and Toloi [stoffer_1992]_ extended the Ljung-Box test to also work
    with missing data. The test statistic is computed as follows:

    .. math ::
        Q_k = n^2 \\sum_{k=1}^{h} \\frac{\\hat{\\rho}_k^2}{n-k}

    where :math:`\\hat{\\rho}_k` is the autocorrelation for lag $k$. When the
    residual time series have non-equidistant time steps it is recommended to
    use this test over the original Ljung-Box test.

    The Stoffer-Toloi test is strictly an adapted version of the Ljung-Box
    test to deal with missing data in a time series and not a time series with
    non-equidistant time steps. This means that the time series is updated
    to an equidistant time series by filling nan-values.

    **Considerations for this test:**

    - Test is also applicable to irregular time series.
    - The time step has to be chosen (e.g., Days). This should not be smaller
      than the smallest time step or the test will most likely fail to
      reject $H_0$ anyway.

    References
    ----------
    .. [stoffer_1992] Stoffer, D. S., & Toloi, C. M. (1992). A note on the
       Ljung—Box—Pierce stoffer_toloi statistic with missing data. Statistics &
       probability letters, 13(5), 391-396.

    Examples
    --------
    >>> data= pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> stat, p = ps.stats.stoffer_toloi(noise, lags=15, freq="D")
    >>> if p > alpha:
    >>>    print("Failed to reject the Null-hypothesis, no significant"
    >>>          "autocorrelation. p =", p.round(2))
    >>> else:
    >>>    print("Reject the Null-hypothesis")

    See Also
    --------
    pastas.utils.get_equidistant_series
    """
    if snap_to_equidistant_timestamps:
        # create equidistant timeseries snapping values from the original
        # series to the nearest equidistant timestamp. No values
        # are duplicated and data loss is minimized.
        s = get_equidistant_series(series, freq, minimize_data_loss=True)
    else:
        # get equidistant sample from original timeseries, checks which
        # time offset is the most common to maximize the number of values
        # taken from the original series.
        t_offset = _get_time_offset(series.index, freq).value_counts().idxmax()
        new_idx = date_range(
            series.index[0].floor(freq) + t_offset,
            series.index[-1].floor(freq) + t_offset,
            freq=freq
        )
        s = series.reindex(new_idx)
        # warn if more than 10% of data is lost in sample
        if s.dropna().index.size < (0.9 * series.dropna().index.size):
            msg = ("While selecting equidistant values from series with "
                   "`as_freq` more than 10 %% of values were dropped. "
                   "Consider setting `make_equidistant` to True."
                   )
            logger.warning(msg)

    nobs = s.size
    z = (s - s.mean()).fillna(0.0)
    y = z.to_numpy()
    yn = s.notna().to_numpy()

    dz0 = (y ** 2).sum() / nobs
    da0 = (yn ** 2).sum() / nobs
    de0 = dz0 / da0

    # initialize, compute all correlation up to one year.
    nlags = 365  # Hard-coded for now
    dz = zeros(nlags)
    da = zeros(nlags)
    de = zeros(nlags)

    for i in range(0, nlags):
        hh = y[:-i - 1] * y[i + 1:]
        dz[i] = hh.sum() / nobs
        hh = yn[:-i - 1] * yn[i + 1:]
        da[i] = hh.sum() / (nobs - i - 1)
        if abs(da[i]) > finfo(float).eps:
            de[i] = dz[i] / da[i]

    re = de / de0

    # remove correlation where no observations are available (de = 0)
    da = da[re != 0][:lags]
    re = re[re != 0][:lags]
    k = arange(1, len(re) + 1)

    # Compute the Q-statistic
    qm = nobs ** 2 * sum(da * re ** 2 / (nobs - k))

    dof = max(len(k) - nparam, 1)
    pval = chi2.sf(qm, df=dof)

    return qm, pval


def diagnostics(series, alpha=0.05, nparam=0, lags=15, stats=(),
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
        DataFrame with the information for the diagnostics checks. The final
        column in this DataFrame report whether or not the Null-Hypothesis
        is rejected. If H0 is not rejected (=False) the data is in agreement
        with one of the properties of white noise (e.g., normally distributed).

    Notes
    -----
    Different tests are computed depending on the regularity of the time
    step of the provided time series. pd.infer_freq is used to
    determined whether or not the time steps are regular.

    Examples
    --------
    >>> data = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> ps.stats.diagnostics(data)
    Out[0]:
                      Checks Statistic P-value  Reject H0
    Shapiroo       Normality      1.00    0.86      False
    D'Agostino     Normality      1.18    0.56      False
    Runs test      Autocorr.     -0.76    0.45      False
    Durbin-Watson  Autocorr.      2.02     nan      False
    Ljung-Box      Autocorr.      5.67    1.00      False

    In this example, the Null-hypothesis is not rejected and the data may be
    assumed to be white noise.
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
    if infer_freq(series.index):
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

    df["Reject H0 ($\\alpha$={:.2f})".format(alpha)] = \
        df.loc[:, "P-value"] < alpha
    df[["Statistic", "P-value"]] = \
        df[["Statistic", "P-value"]].applymap(float_fmt.format)

    return df


def plot_acf():
    raise DeprecationWarning("The method plot_acf is deprecated. Use "
                             "'ps.plot.acf' instead.")


def plot_diagnostics():
    raise DeprecationWarning("The method plot_diagnostics is deprecated."
                             " Use 'ps.plot.diagnostics' instead.")


def plot_cum_frequency():
    raise DeprecationWarning("The method plot_cum_frequency is "
                             "deprecated. Use 'ps.plot.cum_frequency' "
                             "instead.")
