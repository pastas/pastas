"""This module contains utility functions for plotting."""

import logging
from typing import List, Union

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


def _get_height_ratios(ylims: List[Union[list, tuple]]) -> List[float]:
    height_ratios = []
    for ylim in ylims:
        hr = ylim[1] - ylim[0]
        if np.isnan(hr):
            hr = 0.0
        height_ratios.append(hr)
    return height_ratios


def _get_stress_series(ml, split: bool = True) -> List[Series]:
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


def share_xaxes(axes: List[Axes]) -> None:
    """share x-axes"""
    for i, iax in enumerate(axes):
        if i < (len(axes) - 1):
            iax.sharex(axes[-1])
            for t in iax.get_xticklabels():
                t.set_visible(False)


def share_yaxes(axes: List[Axes]) -> None:
    """share y-axes"""
    for iax in axes[1:]:
        iax.sharey(axes[0])
        for t in iax.get_yticklabels():
            t.set_visible(False)


def plot_series_with_gaps(
    series: Series, gap: Timedelta | None = None, ax: Axes | None = None, **kwargs
) -> None:
    """Plot a pandas Series with gaps if index difference larger than gap.

    Parameters
    ----------
    series: pd.Series
        The series to plot.
    gap: Timedelta | None
        Minimum percentage of series to be considered as a gap
    ax: Axes | None
        The axes to plot on.

    kwargs: dict
        Additional keyword arguments that are passed to the plot method.
    """
    if ax is None:
        _, ax = plt.subplots()

    if gap is None:
        gapq = np.quantile(series.index.diff().dropna(), 0.95)
        gap = max(gapq, Timedelta(50, unit="D"))

    s_split = (series.index.diff() >= gap).cumsum()

    series.name = kwargs.pop("label") if "label" in kwargs else series.name
    for i, gr in series.groupby(s_split):
        label = None if i > 0 else series.name
        if len(gr) == 1:
            logger.info(
                f"Isolated point found in series {series.name} with gap larger than {gap.days} days"
            )
            c = kwargs.get("color", None)
            c = kwargs.get("c", c)
            ax.scatter(gr.index, gr.values, label=label, marker="_", s=3.0, c=c)
        ax.plot(gr.index, gr.values, label=label, **kwargs)

    return ax
