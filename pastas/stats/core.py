import numpy as np
from pandas import Series, Timedelta


def acf(x, lags=None, bin_width=None, bin_method='rectangle'):
    """Method to calculate the autocorrelation for irregular timesteps.

    Returns
    -------
    C: pandas.Series
        The autocorrelation function for x.

    See Also
    --------
    ps.stats.ccf

    """
    C = ccf(x=x, y=x, lags=lags, bin_width=bin_width, bin_method=bin_method)
    return C


def ccf(x, y, lags=None, bin_width=None, bin_method='rectangle'):
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
    dt_x = x.index.to_series().diff().values / Timedelta(1, "D")
    dt_x[0] = 0.0
    t_x = np.cumsum(dt_x) / dt_x.mean()

    dt_y = y.index.to_series().diff().values / Timedelta(1, "D")
    dt_y[0] = 0.0
    t_y = np.cumsum(dt_y) / dt_y.mean()

    # TODO Deal with gaps in the data when determining dt_mu?
    dt_mu = max(dt_x.mean(), dt_y.mean())
    dt_min = min(dt_x[1:].min(), dt_y[1:].min())

    # Create matrix with time differences
    t1, t2 = np.meshgrid(t_x, t_y)
    t = np.abs(np.subtract(t1, t2))  # absolute values

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()
    y = (y.values - y.values.mean()) / y.values.std()

    # Create matrix for covariances
    # xx, yy = meshgrid(x, y)
    # xy = xx * yy
    xy = np.outer(y, x)

    if lags is None:  # Default lags in Days
        # lags = [0, 1, 14, 28, 180, 365]
        lags = np.unique(np.logspace(-1, 2.563).astype(int)).tolist()

    # Remove lags that cannot be determined because lag < dt_min
    lags = np.array([lag for lag in lags if lag > dt_min or lag is 0]) / dt_mu

    # Select appropriate bin_width, default depend on bin_method
    if bin_width is None:
        # Select one of the standard options.
        bin_width = {"rectangle": 2, "sinc": 1, "gaussian": 4}
        h = 1 / bin_width[bin_method]
    else:
        h = bin_width / dt_mu

    # Select the binning method to calculate the cross-correlation
    if bin_method == "rectangle":
        kernel_func = lambda d: np.less_equal(d, h).astype(int)
    elif bin_method == "gaussian":
        kernel_func = lambda d: np.exp(-d ** 2 / (2 * h ** 2)) / \
                                np.sqrt(2 * np.pi * h)
    elif bin_method == "sinc":
        raise NotImplementedError()
        # b = np.sin(np.pi * h * d) / (np.pi * h * d) / dt.size
    else:
        raise NotImplementedError(
            "bin_method %s is not implemented." % bin_method)

    del x, y, dt_x, dt_y, t1, t2, t_x, t_y

    C = np.zeros_like(lags)

    # TODO this part can probably be vectorized (nope, too much data)
    # Although we would then have to be aware of of memory errors due to the
    # amount of data. Try to get rid of the for loop (or at least partly)
    # tt = t[:, :, np.newaxis]
    # dd = np.abs(np.subtract(tt, lags))

    # Pre-allocate an array to speed up all numpy methods
    d = np.zeros_like(t)

    for i, k in enumerate(lags):
        # Construct the kernel for the lag
        # d = dd[:, :, i]
        np.abs(np.subtract(t, k, out=d), out=d)
        b = kernel_func(d)
        c = np.multiply(xy, b, out=d)  # Element-wise multiplication
        C[i] = np.sum(c) / np.sum(b)
        del b, c
        # C[i] = calc_acf(xy, b) #

    C = C / np.abs(C).max()

    return Series(data=C, index=lags * dt_mu)
