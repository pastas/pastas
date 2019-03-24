import numpy as np
from pandas import Series, Timedelta, DataFrame
import matplotlib.pyplot as plt


def acf(x, lags=None, bin_method='gaussian', bin_width=None, max_gap=np.inf,
        min_obs=10, output="acf"):
    """Method to calculate the autocorrelation function for irregular
    timesteps based on the slotting technique. Different methods (kernels)
    to bin the data are available.

    Parameters
    ----------
    x: pandas.Series
        Pandas Series containig the values to calculate the
        cross-correlation for. The index has to be a Pandas.DatetimeIndex
    lags: numpy.array, optional
        numpy array containing the lags in days for which the
        cross-correlation if calculated. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 30, 61, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    bin_method: str, optional
        method to determine the type of bin. Options are "gaussian" (default),
        sinc and rectangle.
    bin_width: float, optional
        number of days used as the width for the bin to calculate the
        correlation. By default these values are chosed based on the
        bin_method.
    max_gap: float, optional
        Maximum timestep gap in the data. All timesteps above this gap value
        are not used for calculating the average timestep. This can be
        helpfull when there is a large gap in the data that influences the
        average timestep.

    Returns
    -------
    CCF: pandas.Series
        The Cross-correlation function.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    Examples
    --------
    acf = ps.stats.ccf(x, y, bin_method="gaussian")

    """
    C = ccf(x=x, y=x, lags=lags, bin_method=bin_method, bin_width=bin_width,
            max_gap=max_gap, min_obs=min_obs, output=output)
    C.name = "ACF"  # TODO Rename
    return C


def ccf(x, y, lags=None, bin_method='gaussian', bin_width=None,
        max_gap=np.inf, min_obs=10, output="ccf"):
    """Method to calculate the cross-correlation function for irregular
    timesteps based on the slotting technique. Different methods (kernels)
    to bin the data are available.

    Parameters
    ----------
    x, y: pandas.Series
        Pandas Series containig the values to calculate the
        cross-correlation for. The index has to be a Pandas.DatetimeIndex
    lags: numpy.array, optional
        numpy array containing the lags in days for which the
        cross-correlation if calculated. [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        12, 13, 14, 30, 61, 90, 120, 150, 180, 210, 240, 270, 300, 330, 365]
    bin_method: str, optional
        method to determine the type of bin. Options are "gaussian" (default),
        sinc and rectangle.
    bin_width: float, optional
        number of days used as the width for the bin to calculate the
        correlation. By default these values are chosed based on the
        bin_method.
    max_gap: float, optional
        Maximum timestep gap in the data. All timesteps above this gap value
        are not used for calculating the average timestep. This can be
        helpfull when there is a large gap in the data that influences the
        average timestep.

    Returns
    -------
    CCF: pandas.Series
        The Cross-correlation function.

    References
    ----------
    Rehfeld, K., Marwan, N., Heitzig, J., Kurths, J. (2011). Comparison
    of correlation analysis techniques for irregularly sampled time series.
    Nonlinear Processes in Geophysics. 18. 389-404. 10.5194 pg-18-389-2011.

    Examples
    --------
    acf = ps.stats.ccf(x, y, bin_method="gaussian")

    """
    # prepare the time indices for x and y
    dt_x = x.index.to_series().diff().values / Timedelta(1, "D")
    dt_x[0] = 0.0
    dt_x_mu = dt_x[dt_x < max_gap].mean()  # Deal with big gaps if present
    t_x = np.cumsum(dt_x)

    dt_y = y.index.to_series().diff().values / Timedelta(1, "D")
    dt_y[0] = 0.0
    dt_y_mu = dt_y[dt_y < max_gap].mean()
    t_y = np.cumsum(dt_y)

    dt_mu = max(dt_x_mu, dt_y_mu)

    # Create matrix with time differences
    t1, t2 = np.meshgrid(t_x, t_y)
    
    # Do not take absolute value (previous behavior) and set values to nan where t < 0.
    # This means only positive lags can be calculated!
    t = np.subtract(t1, t2)
    t[t<0] = np.nan

    # Normalize the values and create numpy arrays
    x = (x.values - x.values.mean()) / x.values.std()
    y = (y.values - y.values.mean()) / y.values.std()

    # Create matrix for covariances
    xy = np.outer(y, x)

    if lags is None:  # Default lags in Days, log-scale between 0 and 365.
        lags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 30, 61, 90, 120,
                150, 180, 210, 240, 270, 300, 330, 365]

    # Remove lags that cannot be determined because lag < dt_min
    u, i = np.unique(dt_x, return_counts=True)
    dt_x_min = u[Series(i, u).cumsum() >= min_obs][0]
    u, i = np.unique(dt_y, return_counts=True)
    dt_y_min = u[Series(i, u).cumsum() >= min_obs][0]

    dt_min = min(dt_x_min, dt_y_min)
    # dt_min = min(dt_x[1:].min(), dt_y[1:].min())

    lags = np.array([float(lag) for lag in lags if lag >= dt_min or lag == 0])

    # Delete to free memory
    del (x, y, dt_x, dt_y, t1, t2, t_x, t_y)

    # Select appropriate bin_width, default depend on bin_method
    if bin_width is None:
        options = {"rectangle": 0.5, "sinc": 1, "gaussian": 0.25}
        bin_width = np.ones_like(lags) * options[bin_method] * dt_mu
    elif type(bin_width) is float:
        bin_width = np.ones_like(lags)
    else:
        bin_width = [0.5, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5,
                     2, 2, 2, 2, 2, 2, 2, 2]

    # Select the binning method to calculate the cross-correlation
    if bin_method == "rectangle":
        a = np.zeros_like(t, dtype=float)
        kernel_func = lambda d, h: np.less_equal(np.abs(d, out=a), h,
                                                 out=a).astype(int)
    elif bin_method == "gaussian":
        a = np.zeros_like(t, dtype=float)

        def kernel_func(d, h):
            den1 = -2 * h ** 2  # denominator 1
            den2 = np.sqrt(2 * np.pi * h)  # denominator 2
            return np.exp(np.square(d, out=a) / den1, out=a) / den2
    elif bin_method == "sinc":
        kernel_func = lambda d, h: np.sin(np.pi * h * d) / (np.pi * h * d)
    else:
        raise NotImplementedError("bin_method %s is not implemented." %
                                  bin_method)

    # Pre-allocate an array to speed up all numpy methods
    UDCF = np.zeros_like(lags, dtype=float)
    M = np.zeros_like(lags, dtype=float)
    d = np.zeros_like(t, dtype=float)

    for i, k in enumerate(lags):
        # Construct the kernel for the lag
        np.subtract(t, k, out=d)
        h = bin_width[i]
        b = kernel_func(d, h)
        c = np.multiply(xy, b, out=d)  # Element-wise multiplication
        # Use nansum to avoid the NaNs that are now in these matrices
        UDCF[i] = np.nansum(c)
        M[i] = np.nansum(b)

    DCF = UDCF / M
    CCF = Series(data=DCF, index=lags, name="CCF")

    if output == "full":
        CCFstd = np.sqrt((np.cumsum(UDCF) - M * DCF) ** 2) / (M - 1)
        CCF = DataFrame(data={"CCF": CCF.values, "stderr": CCFstd}, index=lags)

    CCF.index.name = "Lags (Days)"
    return CCF
