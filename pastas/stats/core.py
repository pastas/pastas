from numpy import abs, array, sqrt, pi, exp, meshgrid, zeros_like
from pandas import Series, Timedelta


def acf(x, lags=None, bin_width=None, bin_method='rectangle', tmin=None,
        tmax=None):
    """Method to calculate the autocorrelation for irregular timesteps.

    Returns
    -------
    C: pandas.Series
        The autocorrelation function for x.

    See Also
    --------
    ps.stats.ccf

    """
    C = ccf(x=x, y=x, lags=lags, bin_width=bin_width,
            bin_method=bin_method, tmin=tmin, tmax=tmax)

    return C


def ccf(x, y, lags=None, bin_width=None, bin_method='rectangle', tmin=None,
        tmax=None):
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
    bin_width: float

    bin_method: str
        method to determine the type of bin. Optiona are gaussian, sinc and
        rectangle.

    Returns
    -------
    acf: pandas.Series
        autocorrelation function.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    """

    # Normalize the time values
    dt_x = x.index.to_series().diff() / Timedelta(1, "D")
    dt_x[0] = 0.0
    t_x = (dt_x.cumsum() / dt_x.mean()).values

    dt_y = y.index.to_series().diff() / Timedelta(1, "D")
    dt_y[0] = 0.0
    t_y = (dt_y.cumsum() / dt_y.mean()).values

    dt_mu = max(dt_x.mean(), dt_y.mean())

    # Create matrix with time differences
    t1, t2 = meshgrid(t_x, t_y)
    t = t1 - t2

    # Normalize the values
    x = (x.values - x.mean()) / x.std()
    y = (y.values - y.mean()) / y.std()

    # Create matrix for covariances
    xx, yy = meshgrid(x, y)
    xy = xx * yy

    if lags is None:
        lags = [0, 1, 14, 28, 180, 365]  # Default lags in Days

    # Remove lags that cannot be determined because lag < dt_min
    dt_min = min(dt_x.iloc[1:].min(), dt_y.iloc[1:].min())
    lags = [lag for lag in lags if lag > dt_min or lag is 0]

    lags = array(lags) / dt_mu

    # Select appropriate bin_width, default depend on bin_method
    if bin_width is None:
        # Select one of the standard options.
        bin_width = {"rectangle": 2, "sinc": 1, "gaussian": 4}
        h = 1 / bin_width[bin_method]
    else:
        h = bin_width / dt_mu

    C = zeros_like(lags)

    for i, k in enumerate(lags):
        # Construct the kernel for the lag
        d = abs(abs(t) - k)
        if bin_method == "rectangle":
            b = (d <= h) * 1.
        elif bin_method == "gaussian":
            b = exp(-d ** 2 / (2 * h ** 2)) / sqrt(2 * pi * h)
        elif bin_method == "sinc":
            NotImplementedError()
            # b = np.sin(np.pi * h * d) / (np.pi * h * d) / dt.size
        else:
            NotImplementedError(
                "bin_method %s is not implemented." % bin_method)
        c = xy * b
        C[i] = c.sum() / b.sum()
    C = C / abs(C).max()

    C = Series(data=C, index=lags * dt_mu)

    return C
