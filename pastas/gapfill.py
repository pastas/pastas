"""Gapfill module for filling gaps in head time series using Pastas models."""

from numpy import exp
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    Timestamp,
    concat,
    date_range,
    isna,
)
from scipy.stats import norm

from pastas.typing import Model


def gap_fill(
    head: Series,
    ml: Model,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
    sig_level: float | None = 0.05,
    use_noisemodel: bool = True,
    freq: str | None = None,
) -> Series | DataFrame:
    """Fill gaps in a hydraulic head time series using a Pastas model.

    Parameters
    ----------
    head : pandas.Series
        Hydraulic head time series with gaps. The timestamps where the gaps occur
        should be present in the index, but the values should be NaN.
    ml : Model
        Pastas model to simulate the head.
    tmin : pandas.Timestamp | str | None, optional
        Start date of the simulation. If None, uses the start of the head series.
    tmax : pandas.Timestamp | str | None, optional
        End date of the simulation. If None, uses the end of the head series.
    sig_level : float | None, optional
        Significance level for the prediction intervals. 0.05 by default to compute
        the 95% prediction intervals. If None, no prediction intervals are computed.
    use_noisemodel : bool, optional
        Whether to use the noise model for the correction. Default is True.
        Currently only works with a `ps.ArNoiseModel`.
    freq : str | None, optional
        Frequency for the extended date range. If None, tries to infer from the head
        series. Required if head is empty and tmin/tmax are provided.

    Returns
    -------
    pandas.Series | pandas.DataFrame
        Head time series with gaps filled as a pandas.Series if sig_level is None,
        otherwise as a pandas.DataFrame with columns "head_filled", "lower", and
        "upper", representing the filled head and the prediction intervals.

    Notes
    -----
    The gaps in the head time series are filled using the Pastas model simulation.
    If sig_level is specified, prediction intervals are also computed using the noise
    model. This currently only works with a `ps.ArNoiseModel`.

    Example
    -------
    >>> ps.gapfill(head=head.asfreq("D"), ml=ml, tmin="2000-01-01", tmax="2000-12-31")

    Raises
    ------
    ValueError
        Raised if use_noisemodel is True but no noise model is available in the Pastas
        model.

    """
    # Handle empty series first without tmin or tmax provided
    if head.empty and (tmin is None or tmax is None):
        raise ValueError(
            "An empty head series was provided, please provide a tmin and tmax."
        )
    elif head.empty:
        if freq is None:
            raise ValueError(
                "Cannot create a date range for an empty head series without a freq. "
                "Please provide the freq parameter explicitly."
            )
        head = head.reindex(date_range(start=tmin, end=tmax, freq=freq), dtype=float)

    # Determine tmin and tmax first, converting strings to Timestamp
    if tmin is None:
        tmin = head.index[0]
    elif isinstance(tmin, str):
        tmin = Timestamp(tmin)

    if tmax is None:
        tmax = head.index[-1]
    elif isinstance(tmax, str):
        tmax = Timestamp(tmax)

    # Check if we need to extend the series
    if tmin < head.index[0] or tmax > head.index[-1]:
        new_index = _get_new_index(head, tmin=tmin, tmax=tmax, freq=freq)
        # Reindex the series to the new date range, preserving original values
        head_filled = head.reindex(new_index)
    else:
        head_filled = head.copy()

    # Find start and end dates of the gaps
    starts, ends = _get_gap_starts_ends(head_filled)

    sim = ml.simulate(tmin=tmin, tmax=tmax)
    res = ml.residuals(tmin=tmin, tmax=tmax)

    if use_noisemodel:
        if not ml.noisemodel:
            raise ValueError("No noise model available in the Pastas model.")
        else:
            # do not apply tmin/tmax when computing dt so it uses calibration defaults that were used to estimate alpha
            dt = ml.observations().index.to_series().diff().median() / Timedelta("1D")
            noise_var = ml.noise(tmin=tmin, tmax=tmax).var()
    else:
        res_var = res.var()

    if sig_level is not None:
        variances = Series(0.0, index=head_filled.index, dtype=float)

    for start, end in zip(starts, ends):
        # Use the model to compute the filling values
        filling = sim.loc[start:end]

        tindex = head_filled.loc[start:end].index

        if res.loc[:start].empty:
            r0 = Series(0.0, index=[start])
            r0_is_nan = True
        else:
            r0 = res.loc[:start].iloc[[-1]]
            r0_is_nan = False
        if res.loc[end:].empty:
            r1 = Series(0.0, index=[end])
            r1_is_nan = True
        else:
            r1 = res.loc[end:].iloc[[0]]
            r1_is_nan = False

        t0 = r0.index[0]
        t1 = r1.index[0]

        correction = _compute_correction(
            r0,
            r1,
            tindex,
            ml=ml,
            use_noisemodel=use_noisemodel,
            r0_is_nan=r0_is_nan,
            r1_is_nan=r1_is_nan,
        )

        # Fill the actual gap
        head_filled.loc[start:end] = filling + correction

        if sig_level is not None:
            if use_noisemodel:
                var = _compute_prediction_variance(
                    t0=t0,
                    t1=t1,
                    tindex=tindex,
                    ml=ml,
                    dt=dt,
                    noise_var=noise_var,
                    r0_is_nan=r0_is_nan,
                    r1_is_nan=r1_is_nan,
                )
            else:
                var = Series(res_var, index=tindex)

            variances.loc[start:end] = var

    # If uncertainty is added
    if sig_level is not None:
        z = norm.ppf(1 - sig_level / 2)
        upper = head_filled + z * variances**0.5
        lower = head_filled - z * variances**0.5
        data = concat(
            [head_filled, lower, upper],
            axis=1,
        )
        data.columns = ["head_filled", "lower", "upper"]
        return data
    else:
        return head_filled


def _compute_correction(
    r0: Series,
    r1: Series,
    tindex: DatetimeIndex,
    ml: Model,
    use_noisemodel: bool = True,
    r0_is_nan: bool = False,
    r1_is_nan: bool = False,
) -> Series:
    """Compute the correction for filling gaps in the head time series.

    Parameters
    ----------
    r0 : pandas.Series
        Measured head before the gap. Should be a pandas.Series with a single value
        and a timestamp as the index.
    r1 : pandas.Series
        Measured head after the gap. Should be a pandas.Series with a single value
        and a timestamp as the index.
    tindex : pandas.DatetimeIndex
        Timestamps of the gap.
    ml : Model
        Pastas model.
    use_noisemodel : bool, optional
        Whether to use the noise model for the correction. Default is True.
    r0_is_nan : bool, optional
        Whether the residual before the gap (r0) is NaN (e.g., gap at start).
        Default is False.
    r1_is_nan : bool, optional
        Whether the residual after the gap (r1) is NaN (e.g., gap at end).
        Default is False.

    Returns
    -------
    pandas.Series
        Correction values as a pandas.Series with the timestamps of the gap as the
        index.

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
        alpha = ml.parameters.loc["noise_alpha", "optimal"]

        tau = (tindex - t0) / Timedelta("1D")
        L = (t1 - t0) / Timedelta("1D")

        # Compute the expectation of the residuals for the gap
        a = exp(-tau / alpha)
        b = exp(-(L - tau) / alpha)
        c = exp(-L / alpha)

        if r0_is_nan:
            correction = b * r1
        elif r1_is_nan:
            correction = a * r0
        else:
            correction = (a * r0 + b * r1 - c * (a * r1 + b * r0)) / (1 - c**2)

        correction = Series(correction, index=tindex)
    else:
        s = Series({t0: r0, t1: r1})
        correction = s.reindex(tindex).interpolate(method="time")

    return correction


def _compute_prediction_variance(
    t0: Timestamp,
    t1: Timestamp,
    tindex: DatetimeIndex,
    ml: Model,
    noise_var: float,
    dt: float,
    r0_is_nan: bool = False,
    r1_is_nan: bool = False,
) -> Series:
    """Compute the prediction variance for the gap filling.

    Parameters
    ----------
    t0 : pandas.Timestamp
        Timestamp of the last measurement before the gap.
    t1 : pandas.Timestamp
        Timestamp of the first measurement after the gap.
    tindex : pandas.DatetimeIndex
        Timestamps of the gap.
    ml : Model
        Pastas model.
    dt : float
        Time step of the observations in days.
    noise_var : float
        Variance of the noise.
    r0_is_nan : bool, optional
        Whether the residual before the gap (r0) is NaN (e.g., gap at start). Default is False.
    r1_is_nan : bool, optional
        Whether the residual after the gap (r1) is NaN (e.g., gap at end). Default is False.

    Returns
    -------
    pandas.Series
        Prediction variance for the gap as a pandas.Series with the timestamps of the
        gap as the index.

    """
    alpha = ml.parameters.loc["noise_alpha", "optimal"]

    tau = (tindex - t0) / Timedelta("1D")
    L = (t1 - t0) / Timedelta("1D")

    # Compute the expectation of the variance for the gap
    a = exp(-tau / alpha)
    b = exp(-(L - tau) / alpha)
    c = exp(-L / alpha)

    var_res = noise_var / (1 - exp(-2 * dt / alpha))

    # If we don't have both r0 and r1 (gap at start or end),
    # use the variance of the residuals instead of the formula which may give zero
    if r0_is_nan:
        var = var_res * (1 - b**2)
    elif r1_is_nan:
        # Use a reasonable variance estimate when we don't have both residuals
        var = var_res * (1 - a**2)
    else:
        var = var_res * (1 - a**2 - b**2 + c**2) / (1 - c**2)

    return Series(var, index=tindex)


def _get_gap_starts_ends(head: Series) -> tuple[DatetimeIndex, DatetimeIndex]:
    """Identify the start and end indices of gaps in a time series.

    Parameters
    ----------
    head : pandas.Series
        Time series with a pandas.DatetimeIndex that may contain NaN values (gaps).

    Returns
    -------
    tuple[pandas.DatetimeIndex, pandas.DatetimeIndex]
        A tuple containing:
        - starts: pandas.DatetimeIndex of the start timestamps of all gaps
        - ends: pandas.DatetimeIndex of the end timestamps of all gaps

    Notes
    -----
    This is a private helper function used by gap_fill to identify gaps.
    A gap is defined as a contiguous sequence of NaN values in the series.

    Examples
    --------
    >>> head = pandas.Series([1, np.nan, np.nan, 4, 5],
    ...                  index=pandas.date_range('2000-01-01', periods=5, freq='D'))
    >>> starts, ends = _get_gap_starts_ends(head)
    >>> starts
    pandas.DatetimeIndex(['2000-01-02'], dtype='datetime64[ns]', freq=None)
    >>> ends
    pandas.DatetimeIndex(['2000-01-03'], dtype='datetime64[ns]', freq=None)

    """
    # Find start and end dates of the gaps
    diff = head.isna().astype(int).diff()
    starts = diff[diff == 1.0].index

    # If the time series starts with a gap
    if isna(head.iloc[0]):
        starts = starts.insert(0, head.index[0])

    ends = diff[diff.shift(-1) == -1.0].index

    if isna(head.iloc[-1]):
        ends = ends.union([head.index[-1]])

    return starts, ends


def _get_new_index(
    head: Series,
    tmin: Timestamp,
    tmax: Timestamp,
    freq: str | None = None,
) -> DatetimeIndex:
    """Get a new index if tmin and tmax are outside the range of the head series.

    Parameters
    ----------
    head : pandas.Series
        The original head time series with a pandas.DatetimeIndex.
    tmin : pandas.Timestamp or str
        Start date for the extended series.
    tmax : pandas.Timestamp or str
        End date for the extended series.
    freq : str, optional
        Frequency for the extended date range. If None, tries to infer from the head
        series.

    Returns
    -------
    pandas.DatetimeIndex
        The extended date range covering the full tmin-tmax range with the specified frequency.

    Examples
    --------
    >>> head = pandas.Series([1, 2, 3], index=pandas.date_range('2000-01-01', periods=3, freq='D'))
    >>> _get_new_index(head, tmin='1999-12-30', tmax='2000-01-05', freq='D')
    pandas.DatetimeIndex(['1999-12-30', '1999-12-31', '2000-01-01', '2000-01-02',
                   '2000-01-03', '2000-01-04', '2000-01-05'],
                  dtype='datetime64[ns]', freq='D')

    """
    # Determine frequency if not provided
    if freq is None:
        # Try to infer from the head series
        if hasattr(head.index, "freq") and head.index.freq is not None:
            freq = head.index.freq
        else:
            # Try to infer from the first few differences
            if len(head) > 1:
                diffs = head.index.to_series().diff().dropna()
                if len(diffs) > 0:
                    # Use the most common difference as the frequency
                    freq = diffs.mode()[0]
            if freq is None:
                raise ValueError(
                    "Cannot infer frequency from head series. "
                    "Please provide the freq parameter explicitly."
                )

    # Create the new date range
    new_index = date_range(start=tmin, end=tmax, freq=freq, name=head.index.name)

    return new_index
