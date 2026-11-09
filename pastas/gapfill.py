"""Gapfill module for filling gaps in head time series using Pastas models."""

from numpy import exp, isnan
from pandas import Series, Timedelta, concat
from scipy.stats import norm


def gap_fill(head, ml, tmin=None, tmax=None, sig_level=None, use_noisemodel=True):
    """Fill gaps in a hydraulic head time series using a Pastas model.

    Parameters
    ----------
    head : pd.Series
        hydraulic head time series with gaps. The timestamps where the gaps occur
        should be present in the index, but the values should be NaN.
    ml : ps.Model
        Pastas model to simulate the head.
    tmin : str
        Start date of the simulation.
    tmax : str
        End date of the simulation.
    sig_level : float, optional
        Significance level for the prediction intervals. 0.05 by default to compute the
        95% prediction intervals. If None, no prediction intervals are computed.
    use_noisemodel : bool, optional
        Whether to use the noise model for the correction. Default is True.
        Currentlynly works with a `ps.ArNoiseModel`.

    Returns
    -------
    pd.Series or pd.DataFrame
        Head time series with gaps filled as a Pandas Series if sig_level is None,
        otherwise as a Pandas DataFrame with columns "head_filled", "lower", and "upper", representing the filled head and the prediction intervals.

    Notes
    -----
    The gaps in the head time series are filled using the Pastas model simulation.
    If sig_level is specified, prediction intervals are also computed using the noise
    model. This currently only works with a `ps.ArNoiseModel`.

    Example
    -------
    >>> ps.gapfill(head=head.asfreq("D"), ml=ml, tmin="2000-01-01", tmax="2000-12-31")

    """
    # Make a copy of the head to avoid modifying the original data
    head_filled = head.copy()

    # Find start and end dates of the gaps
    diff = head_filled.isna().astype(int).diff()
    starts = diff[diff == 1.0].index

    # If the time series starts with a gap
    if isnan(head_filled.iloc[0]):
        starts = starts.insert(0, head_filled.index[0])

    ends = diff[diff.shift(-1) == -1.0].index

    # Get the simulation and the residuals
    sim = ml.simulate(tmin=tmin, tmax=tmax)
    res = ml.residuals(tmin=tmin, tmax=tmax)

    if sig_level is not None:
        vars = Series(0.0, index=head_filled.index, dtype=float)

    for start, end in zip(starts, ends):
        # Use the model to compute the filling values
        filling = sim.loc[start:end]

        tindex = head_filled.loc[start:end].index
        r0 = res.loc[:start].iloc[[-1]]
        r1 = res.loc[end:].iloc[[0]]
        t0 = r0.index[0]
        t1 = r1.index[0]

        correction = compute_correction(
            r0, r1, tindex, ml=ml, use_noisemodel=use_noisemodel
        )

        # Fill the actual gap
        head_filled.loc[start:end] = filling + correction

        if sig_level is not None:
            var = compute_prediction_variance(
                t0=t0,
                t1=t1,
                tindex=tindex,
                ml=ml,
                use_noisemodel=use_noisemodel,
            )
            vars.loc[start:end] = var

    # If uncertainty is added
    if sig_level is not None:
        z = norm.ppf(1 - sig_level / 2)
        upper = head_filled + z * vars**0.5
        lower = head_filled - z * vars**0.5
        data = concat(
            [head_filled, lower, upper],
            axis=1,
        )
        data.columns = ["head_filled", "lower", "upper"]
        return data
    else:
        return head_filled


def compute_correction(r0, r1, tindex, ml=None, use_noisemodel=True):
    """Compute the correction for filling gaps in the head time series.

    Parameters
    ----------
    r0 : pd.Series
        Measured head before the gap. Should be a Pandas Series with a single value and a timestamp as the index.
    r1 : pd.Series
        Measured head after the gap. Should be a Pandas Series with a single value and a timestamp as the index.
    tindex : pd.DatetimeIndex
        Timestamps of the gap.
    ml : ps.Model, optional
        Pastas model, not used in the current implementation.
    use_noisemodel : bool, optional
        Whether to use the noise model for the correction. Default is True.

    Notes
    -----
    The correction is computed based on the residuals before and after the gap.
    If a noise model is available and use_noisemodel is True, the correction
    takes the noise model into account. Otherwise, a simple linear interpolation
    is used.

    """
    t0 = r0.index[0]
    t1 = r1.index[0]
    r0 = r0.iloc[0]
    r1 = r1.iloc[0]

    if use_noisemodel:
        if not ml.noisemodel:
            raise ValueError("No noise model available in the Pastas model.")

        alpha = ml.parameters.loc["noise_alpha", "optimal"]

        tau = (tindex - t0) / Timedelta("1D")
        L = (t1 - t0) / Timedelta("1D")

        # Compute the expectation of the residuals for the gap
        a = exp(-tau / alpha)
        b = exp(-(L - tau) / alpha)
        c = exp(-L / alpha)

        correction = (a * r0 + b * r1 - c * (a * r1 + b * r0)) / (1 - c**2)
        correction = Series(correction, index=tindex)
    else:
        s = Series(index=tindex)
        s.loc[t0] = r0
        s.loc[t1] = r1
        correction = s.sort_index().interpolate(method="time").loc[tindex]

    return correction


def compute_prediction_variance(t0, t1, tindex, ml=None, use_noisemodel=True):
    """Compute the prediction variance for the gap filling.

    Parameters
    ----------
    t0 : pd.Timestamp
        Timestamp of the last measurement before the gap.
    t1 : pd.Timestamp
        Timestamp of the first measurement after the gap.
    tindex : pd.DatetimeIndex
        Timestamps of the gap.
    ml : ps.Model, optional
        Pastas model, by default None
    use_noisemodel : bool, optional
        Whether to use the noise model for the prediction variance. Default is True.

    Returns
    -------
    pd.Series
        Prediction variance for the gap as a Pandas Series with the timestamps of the
        gap as the index.

    Raises
    ------
    ValueError
        Raised if use_noisemodel is True but no noise model is available in the Pastas
        model.
    """
    if use_noisemodel:
        if not ml.noisemodel:
            raise ValueError("No noise model available in the Pastas model.")

        alpha = ml.parameters.loc["noise_alpha", "optimal"]

        tau = (tindex - t0) / Timedelta("1D")
        L = (t1 - t0) / Timedelta("1D")

        # Compute the expectation of the variance for the gap
        a = exp(-tau / alpha)
        b = exp(-(L - tau) / alpha)
        c = exp(-L / alpha)

        dt = ml.observations().index.to_series().diff().median() / Timedelta("1D")
        var_noise = ml.noise().var()
        var_res = var_noise / (1 - exp(-2 * dt / alpha))

        var = var_res * (1 - a**2 - b**2 + c**2) / (1 - c**2)
    else:
        var = ml.residuals().var()

    return Series(var, index=tindex)
