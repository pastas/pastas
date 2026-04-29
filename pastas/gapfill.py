from numpy import array, exp, isnan
from pandas import DataFrame, Series, Timedelta, concat


def gap_fill(head, ml, tmin, tmax, alpha=None):
    """gap_fill_head _summary_

    Parameters
    ----------
    head : pd.Series
        hydraulic head time series with gaps. The timestamps which are missing should be present in the index, but the values should be NaN.
    ml : ps.Model
        Pastas model to simulate the head.
    tmin : str
        Start date of the simulation.
    tmax : str
        End date of the gap
    alpha : float, optional
        0.05

    Returns
    -------
    pd.Series
        Head time series with gaps filled.

    Example
    -------
    >>> ps.gapfill(head=head.asfreq("D"), ml=ml, tmin="2000-01-01", tmax="2000-12-31")

    """
    # Make a copy of the head to avoid modifying the original data
    head_filled = head.copy()

    # Find start and end dates of the gaps
    diff = head.isna().astype(int).diff()
    starts = diff[diff == 1.0].index.shift(-1, freq="D")

    # If the time series starts with a gap
    if isnan(head.iloc[0]):
        starts = starts.insert(0, head.index[0])

    ends = diff[diff == -1.0].index

    sim = ml.simulate(tmin=tmin, tmax=tmax)

    for start, end in zip(starts, ends):
        # Use the model to compute the filling values
        filling = sim.loc[start:end]
        correction = (filling - head.loc[start:end]).interpolate().fillna(0)

        # Fill the actual gap
        head_filled.loc[start + Timedelta("1D") : end - Timedelta("1D")] = (
            filling.iloc[1:-1] - correction.iloc[1:-1]
        )

    # If uncertainty is added
    if alpha is not None:
        pi = compute_prediction_error(head, head_filled, ml)
        data = concat(
            [head_filled, pi],
            axis=1,
        )
        data.columns = ["head_filled", "lower", "upper"]
        return data
    else:
        return head


def compute_prediction_error(head, head_filled, ml):
    # Drop the nas-values
    head = head.dropna()

    # std_res = ml.residuals().std()
    if ml.noisemodel:
        std_noise = ml.noise().std()

        a = ml.parameters.loc["noise_alpha", "optimal"]
        dt = 10  # ml.settings("freq_obs") ??

        diff = array(
            [
                abs(head_filled.index[i] - head.index).min()
                for i in range(len(head_filled))
            ]
        ) / Timedelta("1D")
        phi = exp(-1 / a * dt)
        e = std_noise * (1 - phi ** (2 * diff)) / (1 - phi**2)
        error = Series(index=head_filled.index, data=e)

        z = 1.96

        pi = DataFrame(
            data={"lower": head_filled - z * error, "upper": head_filled + z * error}
        )

        return pi
    else:
        return None
