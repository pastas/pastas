from numpy import isnan
from pandas import Timedelta


def gap_fill(head, ml, tmin, tmax):
    """gap_fill_head _summary_

    Parameters
    ----------
    head : pd.Series
        hydraulic head time series with gaps.
    ml : ps.Model
        Pastas model to simulate the head.
    tmin : str
        Start date of the simulation.
    tmax : str
        End date of the simulation.

    Returns
    -------
    pd.Series
        Head time series with gaps filled.

    """
    head = head.copy()
    # Find start and end dates of the gaps
    diff = head.isna().astype(int).diff()
    starts = diff[diff == 1.0].index.shift(-1, freq="D")

    if isnan(head.iloc[0]):
        starts = starts.insert(0, head.index[0])

    ends = diff[diff == -1.0].index

    sim = ml.simulate(tmin=tmin, tmax=tmax)

    for start, end in zip(starts, ends):
        # Use the model to compute the filling values
        filling = sim.loc[start:end]
        correction = (filling - head.loc[start:end]).interpolate().fillna(0)

        # Fill the actual gap
        head.loc[start + Timedelta("1D") : end - Timedelta("1D")] = (
            filling.iloc[1:-1] - correction.iloc[1:-1]
        )

    return head


