"""This module contains utility functions for plotting."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from pandas import Series, Timedelta

from pastas.typing import Axes

logger = logging.getLogger(__name__)


def _table_formatter_params(s: float, na_rep: str = "") -> str:
    """Internal method for formatting parameters in tables in Pastas plots.

    Parameters
    ----------
    s : float
        value to format.

    Returns
    -------
    str
        float formatted as str.
    """
    if np.isnan(s):
        return na_rep
    elif np.floor(np.log10(np.abs(s))) <= -2:
        return f"{s:.2e}"
    elif np.floor(np.log10(np.abs(s))) > 5:
        return f"{s:.2e}"
    else:
        return f"{s:.2f}"


def _table_formatter_stderr(s: float, na_rep: str = "") -> str:
    """Internal method for formatting stderrs in tables in Pastas plots.

    Parameters
    ----------
    s : float
        value to format.

    Returns
    -------
    str
        float formatted as str.
    """
    if np.isnan(s):
        return na_rep
    elif s == 0.0:
        return f"±{s * 100:.2e}%"
    elif np.floor(np.log10(np.abs(s))) <= -4:
        return f"±{s * 100.0:.2e}%"
    elif np.floor(np.log10(np.abs(s))) > 3:
        return f"±{s * 100.0:.2e}%"
    else:
        return f"±{s:.2%}"


def _get_height_ratios(ylims: list[tuple[float, float]]) -> list[float]:
    return [0.0 if np.isnan(ylim[1] - ylim[0]) else ylim[1] - ylim[0] for ylim in ylims]


def _get_stress_series(ml, split: bool = True) -> list[Series]:
    stresses = []
    for name in ml.stressmodels.keys():
        nstress = len(ml.stressmodels[name].stress)
        if split and nstress > 1:
            for istress in range(nstress):
                stress = ml.get_stress(name, istress=istress)
                stresses.append(stress)
        else:
            stress = ml.get_stress(name)
            if isinstance(stress, list):
                stresses.extend(stress)
            else:
                stresses.append(stress)
    return stresses


def share_xaxes(axes: list[Axes]) -> None:
    """share x-axes"""
    for i, iax in enumerate(axes):
        if i < (len(axes) - 1):
            iax.sharex(axes[-1])
            for t in iax.get_xticklabels():
                t.set_visible(False)


def share_yaxes(axes: list[Axes]) -> None:
    """share y-axes"""
    for iax in axes[1:]:
        iax.sharey(axes[0])
        for t in iax.get_yticklabels():
            t.set_visible(False)


def plot_series_with_gaps(
    series: Series, gap: Timedelta | None = None, ax: Axes | None = None, **kwargs
) -> Axes:
    """Plot a pandas Series with gaps if index difference is larger than gap.

    Parameters
    ----------
    series: pd.Series
        The series to plot.
    gap: Timedelta | None
        Timedelta to be considered as a gap. If the difference between two
        consecutive index values is larger than gap, a gap is inserted in the
        plot. If None, the maximum value between the 95th percentile of the
        differences and 50 days is used as gap.
    ax: Axes | None
        The axes to plot on. if None, a new figure is created.
    kwargs: dict
        Additional keyword arguments that are passed to the plot method.
    """
    if ax is None:
        _, ax = plt.subplots()

    td_diff = series.index[1:] - series.index[:-1]
    if gap is None:
        gapq = np.quantile(td_diff, 0.95)
        gap = max(gapq, Timedelta(50, unit="D"))

    s_split = np.append(0.0, np.cumsum(td_diff >= gap))

    series.name = kwargs.pop("label") if "label" in kwargs else series.name
    color = kwargs.pop("c", "k")
    color = kwargs.pop("color", color)
    for i, gr in series.groupby(s_split):
        label = None if i > 0 else series.name
        if len(gr) == 1:
            logger.info(
                "Isolated point found in series %s with gap larger than %s days",
                series.name,
                gap / Timedelta(1, "D"),
            )
            ax.scatter(gr.index, gr.values, label=label, marker="_", s=3.0, color=color)
        ax.plot(gr.index, gr.values, label=label, color=color, **kwargs)

    return ax
