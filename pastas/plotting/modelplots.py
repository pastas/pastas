"""This module contains plotting methods for Pastas Models."""

import logging
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogFormatter, MultipleLocator
from pandas import Series, Timestamp, concat

from pastas.decorators import PastasDeprecationWarning, model_tmin_tmax
from pastas.plotting.plots import cum_frequency, diagnostics, pairplot, series
from pastas.plotting.plotutil import (
    _get_height_ratios,
    _get_stress_series,
    _table_formatter_params,
    _table_formatter_stderr,
    plot_series_with_gaps,
    share_xaxes,
)
from pastas.rfunc import HantushWellModel
from pastas.stressmodels import ChangeModel, TarsoModel
from pastas.timeseries_utils import _get_dt
from pastas.typing import Axes, Figure, Model, TimestampType

logger = logging.getLogger(__name__)


class Plotting:
    """Class that contains all plotting methods for Pastas models.

    Pastas models come with a number of predefined plotting methods to quickly
    visualize a Model. All of these methods are contained in the `plot` attribute of
    a model. For example, if we stored a :class:`pastas.model.Model` instance in the
    variable `ml`, the plot methods are available as follows::

    >>> ml.plots.results()
    """

    def __init__(self, ml: Model) -> None:
        self.ml = ml  # Store a reference to the model class

    def __repr__(self) -> str:
        msg = (
            "This module contains all the built-in plotting options that are available."
        )
        return msg

    @model_tmin_tmax
    def plot(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        oseries: bool = True,
        simulation: bool = True,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        oseries: bool, optional
            True to plot the observed time series.
        simulation: bool, optional
            True to plot the simulated time series.
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        legend: bool, optional
            Boolean to determine to show the legend (True) or not (False).

        Returns
        -------
        ax: matplotlib.axes.Axes
            matplotlib axes with the simulated and optionally the observed time series.

        Examples
        --------
        >>> ml.plot()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, **kwargs)

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o_nu = self.ml.oseries.series.drop(o.index).loc[
                o.index.min() : o.index.max()
            ]
            if not o_nu.empty:
                # plot parts of the oseries that are not used in grey
                o_nu.plot(linestyle="", marker=".", color="0.5", label="", ax=ax)
            o.plot(linestyle="", marker=".", color="k", ax=ax)

        if simulation:
            sim = self.ml.simulate(tmin=tmin, tmax=tmax)
            r2 = self.ml.stats.rsq(tmin=tmin, tmax=tmax)
            sim.plot(ax=ax, label=f"{sim.name} ($R^2$={r2:.2%})")

        # Dress up the plot
        # temporary fix, as set_xlim currently does not work with strings mpl=3.6.1
        if tmin is not None:
            tmin = Timestamp(tmin)
        if tmax is not None:
            tmax = Timestamp(tmax)

        ax.set_xlim(tmin, tmax)
        ax.set_ylabel("Head")

        if legend:
            ax.legend(ncol=2, numpoints=3)
        plt.tight_layout()
        return ax

    @model_tmin_tmax
    def results(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        figsize: tuple = (10, 8),
        split: bool = False,
        adjust_height: bool = True,
        return_warmup: bool = False,
        block_or_step: str = "step",
        stderr: bool = False,
        fig: Figure | None = None,
        **kwargs,
    ) -> Axes:
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        split: bool, optional
            Split the stresses in multiple stresses when possible. Default is False.
        adjust_height: bool, optional
            Adjust the height of the graphs, so that the vertical scale of all the
            subplots on the left is equal. Default is True.
        return_warmup: bool, optional
            Show the warmup-period. Default is false.
        block_or_step: str, optional
            Plot the block- or step-response on the right. Default is 'step'.
        stderr : bool, optional
            If True the standard error of the parameter values are shown. Please be
            aware of the conditions for reliable uncertainty estimates, more
            information here:
            https://pastas.readthedocs.io/stable/examples/diagnostic_checking.html
        fig: matplotib.Figure instance, optional
            Optionally provide a matplotib.Figure instance to plot onto.
        **kwargs: dict, optional
            Optional arguments, passed on to the matplotlib.pyplot.figure method.

        Returns
        -------
        list of matplotlib.axes.Axes

        Examples
        --------
        >>> ml.plots.results()
        """
        # Number of rows to make the figure with
        o = self.ml.observations(tmin=tmin, tmax=tmax)
        o_nu = self.ml.oseries.series.drop(o.index)
        if return_warmup:
            o_nu = o_nu[tmin - self.ml.settings["warmup"] : tmax]
        else:
            o_nu = o_nu[tmin:tmax]
        sim = self.ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        contribs = self.ml.get_contributions(
            split=split, tmin=tmin, tmax=tmax, return_warmup=return_warmup
        )

        ylims = [
            (
                min([sim.min(), o[tmin:tmax].min()]),
                max([sim.max(), o[tmin:tmax].max()]),
            ),
            (res.min(), res.max()),
        ]  # residuals are bigger than noise

        if adjust_height:
            for contrib in contribs:
                hs = contrib.loc[tmin:tmax]
                if hs.empty:
                    if contrib.empty:
                        ylims.append((0.0, 0.0))
                    else:
                        ylims.append((contrib.min(), hs.max()))
                else:
                    ylims.append((hs.min(), hs.max()))
            hrs = _get_height_ratios(ylims)
        else:
            hrs = [2] + [1] * (len(contribs) + 1)

        # Make main Figure
        if fig is None:
            fig = plt.figure(figsize=figsize, **kwargs)

        gs = fig.add_gridspec(
            ncols=2, nrows=len(contribs) + 2, width_ratios=[2, 1], height_ratios=hrs
        )

        # Main frame
        ax1 = fig.add_subplot(gs[0, 0])
        o.plot(ax=ax1, linestyle="", marker=".", color="k", x_compat=True)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(
                ax=ax1,
                linestyle="",
                marker=".",
                color="0.5",
                label="",
                x_compat=True,
                zorder=-1,
            )

        # add rsq to simulation
        r2 = self.ml.stats.rsq(tmin=tmin, tmax=tmax)
        sim.plot(ax=ax1, x_compat=True, label=f"{sim.name} ($R^2$={r2:.2%})")
        ax1.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)
        ax1.set_ylim(ylims[0])
        ax1.set_ylabel("Head")

        # Residuals and noise
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax2 = plot_series_with_gaps(res, ax=ax2, color="k")
        if self.ml.settings["noise"] and self.ml.noisemodel:
            noise = self.ml.noise(tmin=tmin, tmax=tmax)
            ax2 = plot_series_with_gaps(noise, ax=ax2, color="C0")
        ax2.axhline(0.0, color="k", linestyle="--", zorder=0)
        ax2.legend(loc=(0, 1), ncol=3, frameon=False)

        # Add a row for each stressmodel
        rmax = 0.0  # tmax of the response
        ax_response = None
        i = 0
        for sm_name, sm in self.ml.stressmodels.items():
            # plot the contribution
            nsplit = sm.get_nsplit() if split else 1
            for istress in range(nsplit):
                ax_contrib = fig.add_subplot(gs[i + 2, 0], sharex=ax1)
                contribs[i].plot(ax=ax_contrib, x_compat=True)
                ax_contrib.legend(loc=(0, 1), ncol=3, frameon=False)
                ax_contrib.set_ylabel("Rise")
                if adjust_height:
                    ax_contrib.set_ylim(ylims[i + 2])
                if not split:
                    title = [stress.name for stress in sm.stress]
                    if len(title) > 3:
                        title = title[:3] + ["..."]
                    ax_contrib.set_title(
                        f"Stresses: {title}",
                        loc="right",
                        fontsize=plt.rcParams["legend.fontsize"],
                    )

                ax_response = gs.figure.add_subplot(gs[i + 2, 1], sharex=ax_response)
                ax_response = self._plot_response_in_results(
                    sm_name=sm_name,
                    block_or_step=block_or_step,
                    ax=ax_response,
                    istress=istress if split else None,
                )
                ax_response_xlim = ax_response.get_xlim()
                rmax = max(rmax, ax_response_xlim[1])
                ax_response.set_xlim(left=ax_response_xlim[0], right=rmax)
                ax_response.set_title(
                    f"{block_or_step.capitalize()} response",
                    fontsize=plt.rcParams["legend.fontsize"],
                )
                i += 1

        # xlim sets minorticks back after plots:
        ax1.minorticks_off()

        # temporary fix, as set_xlim currently does not work with strings mpl=3.6.1
        if tmin is not None:
            tmin = Timestamp(tmin)
        if tmax is not None:
            tmax = Timestamp(tmax)

        if return_warmup:
            ax1.set_xlim(tmin - self.ml.settings["warmup"], tmax)
        else:
            ax1.set_xlim(tmin, tmax)

        # sometimes, ticks suddenly appear on top plot, turn off just in case
        plt.setp(ax1.get_xticklabels(), visible=False)

        for ax in fig.axes:
            ax.grid(True)

        if isinstance(fig, plt.Figure):
            fig.tight_layout(pad=0.0)  # before making the table

        # plot parameters table
        ax3 = fig.add_subplot(gs[0:2, 1])
        _ = self._plot_parameters_table(ax=ax3, stderr=stderr)

        return fig.axes

    @model_tmin_tmax
    def results_mosaic(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        stderr: bool = False,
        block_or_step: str = "step",
        return_warmup: bool = False,
        adjust_height: bool = True,
        figsize: tuple[float, float] | None = None,
        layout: Literal["constrained", "tight", "compressed", "none"]
        | None = "constrained",
        fig_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Axes]:
        """Plot the results of the model in a mosaic plot.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        stderr : bool, optional
            If True the standard error of the parameter values are shown.
        block_or_step: str, optional
            Plot the block- or step-response on the right. Default is 'step'.
        adjust_height: bool, optional
            Adjust the height of the graphs, so that the vertical scale of all the
            subplots on the left is equal. Default is True.
        return_warmup: bool, optional
            Show the warmup-period. Default is False.
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.

        **kwargs: dict, optional
            Optional arguments, passed on to the matplotlib.pyplot.figure method.

        Returns
        -------
        Dictionary with the matplotlib.axes.Axes

        Examples
        --------
        >>> ml.plots.results_mosaic()
        """

        tmin = Timestamp(tmin) if tmin is not None else None
        tmax = Timestamp(tmax) if tmax is not None else None

        # get simulated time series
        o = self.ml.observations(tmin=tmin, tmax=tmax)
        o_nu = self.ml.oseries.series.drop(o.index)
        o_nu = (
            o_nu[tmin - self.ml.settings["warmup"] : tmax]
            if return_warmup
            else o_nu[tmin:tmax]
        )
        sim = self.ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        contribs = {
            x.name: x
            for x in self.ml.get_contributions(
                tmin=tmin,
                tmax=tmax,
                return_warmup=return_warmup,
                split=False,
            )
        }

        # setup ylims
        ylims = {
            "sim": [
                min([sim.min(), o[tmin:tmax].min(), o_nu.min()]),
                max([sim.max(), o[tmin:tmax].max(), o_nu.max()]),
            ],
            "res": [res.min(), res.max()],
        }
        for k, ylim in ylims.items():
            yl_diff = (ylim[1] - ylim[0]) * 0.025
            ylims[k] = [ylim[0] - yl_diff, ylim[1] + yl_diff]

        for cname, contrib in contribs.items():
            hs = contrib.loc[tmin:tmax]
            if hs.empty:
                if contrib.empty:
                    ylim_c = [0.0, 0.0]
                else:
                    ylim_c = [contrib.min(), hs.max()]
            else:
                ylim_c = [hs.min(), hs.max()]
            ylims[f"con_{cname}"] = ylim_c

        # construct mosoaic
        mosaic = [[x] for x in ylims]
        for mos in mosaic:
            if "con_" in mos[0]:
                mos.append(f"rf_{mos[0].split('_', 1)[1]}")
            elif mos[0] in ("sim", "res"):
                mos.append("tab")

        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        if "width_ratios" not in fig_kwargs:
            fig_kwargs["width_ratios"] = [2.0, 1.0]
        height_ratios = (
            _get_height_ratios(list(ylims.values()))
            if adjust_height
            else fig_kwargs.pop("height_ratios", None)
        )

        figsize = (10, 4 + 2 * len(contribs)) if figsize is None else figsize
        _, axd = plt.subplot_mosaic(
            mosaic,
            height_ratios=height_ratios,
            layout=layout,
            figsize=figsize,
            **fig_kwargs,
        )

        # plot observations and simulation
        axd["sim"].plot(
            o.index, o.values, linestyle="", marker=".", color="k", label=o.name
        )
        if not o_nu.empty:
            axd["sim"].plot(
                o_nu.index,
                o_nu.values,
                linestyle="",
                marker=".",
                color="grey",
                label="",
                zorder=-1,
            )
        axd["sim"].plot(
            sim.index,
            sim.values,
            label=f"{sim.name} ($R^2$={self.ml.stats.rsq(tmin=tmin, tmax=tmax):.2%})",
        )
        axd["sim"].legend(loc=(0, 1), ncol=2, frameon=False, numpoints=3)
        axd["sim"].set_ylim(bottom=ylims["sim"][0], top=ylims["sim"][1])

        # plot residuals (and noise if present)
        _ = plot_series_with_gaps(res, ax=axd["res"], color="k")
        if self.ml.settings["noise"] and self.ml.noisemodel:
            noise = self.ml.noise(tmin=tmin, tmax=tmax)
            _ = plot_series_with_gaps(noise, ax=axd["res"], color="C0")
        axd["res"].axhline(0.0, color="k", linestyle="--", zorder=0)
        axd["res"].legend(loc=(0, 1), ncol=2, frameon=False)

        # plot the contributions and responses of the stressmodels
        for sm_name, sm in self.ml.stressmodels.items():
            axd[f"con_{sm_name}"].plot(
                contribs[sm_name].index,
                contribs[sm_name].values,
                label=sm_name,
            )
            title = [stress.name for stress in sm.stress]
            if len(title) > 3:
                title = title[:3] + ["..."]
            if title:
                axd[f"con_{sm_name}"].set_title(
                    "Stresses: " + str(title).replace("'", ""),
                    loc="right",
                    fontsize=plt.rcParams["legend.fontsize"],
                )
            axd[f"con_{sm_name}"].legend(loc=(0, 1), ncol=1, frameon=False)
            axd[f"con_{sm_name}"].set_ylim(ylims[f"con_{sm_name}"])
            _ = self._plot_response_in_results(
                sm_name=sm_name,
                block_or_step=block_or_step,
                ax=axd[f"rf_{sm_name}"],
            )

        # share x-axes of simulation, residuals and contributions
        share_xaxes([axd[k] for k in [x[0] for x in mosaic]])
        axd["sim"].set_xlim(
            tmin - self.ml.settings["warmup"], tmax
        ) if return_warmup else axd["sim"].set_xlim(tmin, tmax)

        # add legend to the upper response axes and share x-axes of responses
        response_axes = [axd[k] for k in [x[1] for x in mosaic] if k.startswith("rf_")]
        response_axes[0].legend(loc=(0, 1), frameon=False)

        response_xlims = [ax.get_xlim() for ax in response_axes]
        share_xaxes(response_axes)
        response_axes[-1].set_xlim(
            left=min(x[0] for x in response_xlims),
            right=max(x[1] for x in response_xlims),
        )

        for k in axd:
            axd[k].grid(True)
            axd[k].yaxis.tick_right() if k.startswith("rf_") else axd[
                k
            ].yaxis.tick_left()

        _ = self._plot_parameters_table(ax=axd["tab"], stderr=stderr)

        return axd

    def _plot_response_in_results(
        self,
        sm_name: str,
        block_or_step: Literal["step", "block"],
        ax: Axes,
        istress: int | None = None,
    ):
        """Internal method to plot the response of a Stressmodel in the results-plot"""
        rkwargs = {}
        sm = self.ml.stressmodels[sm_name]
        if isinstance(sm, (ChangeModel, TarsoModel)):
            dt = _get_dt(self.ml.settings["freq"])
            if isinstance(sm, ChangeModel):
                parnames0 = [
                    x.split("_")
                    for x in list(sm.rfunc1.get_init_parameters(sm_name).index)
                ]
                response0 = getattr(sm.rfunc1, block_or_step)(
                    p=self.ml.parameters.loc[
                        [f"{x[0]}_1_{x[1]}" for x in parnames0], "optimal"
                    ].values,
                    dt=dt,
                )
                parnames1 = [
                    x.split("_")
                    for x in list(sm.rfunc2.get_init_parameters(sm_name).index)
                ]
                response1 = getattr(sm.rfunc2, block_or_step)(
                    p=self.ml.parameters.loc[
                        [f"{x[0]}_2_{x[1]}" for x in parnames1], "optimal"
                    ].values,
                    dt=dt,
                )
            elif isinstance(sm, TarsoModel):
                parnames = list(sm.rfunc.get_init_parameters(sm_name).index)
                response0 = getattr(sm.rfunc, block_or_step)(
                    p=self.ml.parameters.loc[
                        [f"{x}0" for x in parnames], "optimal"
                    ].values,
                    dt=dt,
                )
                response1 = getattr(sm.rfunc, block_or_step)(
                    p=self.ml.parameters.loc[
                        [f"{x}1" for x in parnames], "optimal"
                    ].values,
                    dt=dt,
                )
            responses = [
                Series(
                    np.insert(response, 0, 0.0),
                    index=np.linspace(0, response.size * dt, response.size + 1),
                    name=f"{sm_name}_rf{i}",
                )
                for i, response in enumerate([response0, response1])
            ]
        else:
            if isinstance(sm.rfunc, HantushWellModel):
                rkwargs = {"warn": False}
                # show the response of the first well, which gives more information than istress = None
                istress = 0 if istress is None else istress
            responses = [
                self.ml._get_response(
                    block_or_step=block_or_step,
                    name=sm_name,
                    add_0=True,
                    istress=istress,
                    **rkwargs,
                )
            ]

        responses = [x for x in responses if x is not None]
        if responses:
            xlim_left = min(
                [
                    x.index[0] if block_or_step == "step" else x.index[1]
                    for x in responses
                    if x is not None
                ]
            )
            xlim_right = max([x.index[-1] for x in responses])
            for i, response in enumerate(responses):
                if i == 0:
                    label = f"{block_or_step.capitalize()} response"
                    if block_or_step == "block":
                        ax.set_xscale("log")
                        ax.xaxis.set_major_formatter(LogFormatter())
                else:
                    label = None
                ax.plot(response.index, response.values, label=label)
                ax.set_xlim(left=xlim_left, right=xlim_right)
        return ax

    def _plot_parameters_table(self, ax: Axes, stderr: bool) -> None:
        """Internal method to plot the parameters table in the results-plot"""
        ax.set_title(
            f"Model parameters ($n_c$={self.ml.parameters.vary.sum()})",
            loc="left",
            fontsize=plt.rcParams["legend.fontsize"],
        )
        p = self.ml.parameters.loc[:, ["name"]].copy()
        p.loc[:, "name"] = p.index
        p.loc[:, "optimal"] = self.ml.parameters.loc[:, "optimal"].apply(
            _table_formatter_params
        )
        if stderr:
            stderrper = (
                self.ml.parameters.loc[:, "stderr"]
                / self.ml.parameters.loc[:, "optimal"]
            )
            p.loc[:, "stderr"] = stderrper.abs().apply(_table_formatter_stderr)
        ax.axis("off")
        ax.table(
            bbox=(0.0, 0.0, 1.0, 1.0),
            cellText=p.values,
            colWidths=[p[col].str.len().max() for col in p.columns],
            colLabels=p.columns,
        )
        return ax

    @model_tmin_tmax
    def decomposition(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        ytick_base: bool = True,
        split: bool = True,
        figsize: tuple = (10, 8),
        axes: Axes | None = None,
        name: str | None = None,
        return_warmup: bool = False,
        min_ylim_diff: float | None = None,
        **kwargs,
    ) -> Axes:
        """Plot the decomposition of a time-series in the different stresses.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        ytick_base: Boolean or float, optional
            Make the ytick-base constant if True, set this base to float if a float.
        split: bool, optional
            Split the stresses in multiple stresses when possible. Default is True.
        axes: matplotlib.axes.Axes instance, optional
            Matplotlib Axes instance to plot the figure on to.
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        name: str, optional
            Name to give the simulated time series in the legend.
        return_warmup: bool, optional
            Show the warmup-period. Default is false.
        min_ylim_diff: float, optional
            Float with the difference in the ylimits. Default is None
        **kwargs: dict, optional
            Optional arguments, passed on to the matplotlib.pyplot.subplots method.

        Returns
        -------
        axes: list of matplotlib.axes.Axes
        """
        o = self.ml.observations(tmin=tmin, tmax=tmax)

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
        if name is not None:
            sim.name = name

        # determine the influence of the different stresses
        contribs = self.ml.get_contributions(
            split=split, tmin=tmin, tmax=tmax, return_warmup=return_warmup
        )
        names = [s.name for s in contribs]

        if self.ml.transform:
            contrib = self.ml.get_transform_contribution(tmin=tmin, tmax=tmax)
            contribs.append(contrib)
            names.append(self.ml.transform.name)

        # determine ylim for every graph, to scale the height
        ylims = [
            (min([sim.min(), o[tmin:tmax].min()]), max([sim.max(), o[tmin:tmax].max()]))
        ]
        for contrib in contribs:
            hs = contrib[tmin:tmax]
            if hs.empty:
                if contrib.empty:
                    ylims.append((0.0, 0.0))
                else:
                    ylims.append((contrib.min(), hs.max()))
            else:
                ylims.append((hs.min(), hs.max()))
        if min_ylim_diff is not None:
            for i, ylim in enumerate(ylims):
                if np.diff(ylim) < min_ylim_diff:
                    ylims[i] = (
                        np.mean(ylim) - min_ylim_diff / 2,
                        np.mean(ylim) + min_ylim_diff / 2,
                    )
        # determine height ratios
        height_ratios = _get_height_ratios(ylims)

        nrows = len(contribs) + 1
        if axes is None:
            # open a new figure
            gridspec_kw = {"height_ratios": height_ratios}
            fig, axes = plt.subplots(
                nrows, sharex=True, figsize=figsize, gridspec_kw=gridspec_kw, **kwargs
            )
            axes = np.atleast_1d(axes)
            o_label = o.name
            set_axes_properties = True
        else:
            if len(axes) != nrows:
                msg = "Makes sure the number of axes equals the number of series"
                raise Exception(msg)
            fig = axes[0].figure
            o_label = ""
            set_axes_properties = False

        # plot simulation and observations in top graph
        o_nu = self.ml.oseries.series.drop(o.index)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(
                linestyle="",
                marker=".",
                color="0.5",
                label="",
                markersize=2,
                ax=axes[0],
                x_compat=True,
            )
        o.plot(
            linestyle="",
            marker=".",
            color="k",
            label=o_label,
            markersize=3,
            ax=axes[0],
            x_compat=True,
        )

        r2 = self.ml.stats.rsq(tmin=tmin, tmax=tmax)
        sim.plot(ax=axes[0], x_compat=True, label=f"{sim.name} ($R^2$={r2:.2%})")
        if set_axes_properties:
            axes[0].set_ylim(ylims[0])
        axes[0].grid(True)
        axes[0].legend(ncol=3, frameon=False, numpoints=3)
        axes[0].set_ylabel("Head")

        if ytick_base and set_axes_properties:
            if isinstance(ytick_base, bool):
                # determine the ytick-spacing of the top graph
                yticks = axes[0].yaxis.get_ticklocs()
                if len(yticks) > 1:
                    ytick_base = yticks[1] - yticks[0]
                else:
                    ytick_base = None
            axes[0].yaxis.set_major_locator(MultipleLocator(base=ytick_base))

        # plot the influence of the stresses
        for i, contrib in enumerate(contribs):
            ax = axes[i + 1]
            contrib.plot(ax=ax, x_compat=True)
            if set_axes_properties:
                if ytick_base:
                    # set the ytick-spacing equal to the top graph
                    locator = MultipleLocator(base=ytick_base)
                    ax.yaxis.set_major_locator(locator)
                ax.set_title(names[i])
                ax.set_ylim(ylims[i + 1])
            ax.grid(True)
            ax.minorticks_off()
            ax.set_ylabel("Rise")
        if set_axes_properties:
            # temporary fix, as set_xlim currently does not work with strings mpl=3.6.1
            if tmin is not None:
                tmin = Timestamp(tmin)
            if tmax is not None:
                tmax = Timestamp(tmax)
            axes[0].set_xlim(tmin, tmax)
        fig.tight_layout(pad=0.0)

        return axes

    @model_tmin_tmax
    def diagnostics(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        figsize: tuple = (10, 5),
        bins: int = 50,
        acf_options: dict | None = None,
        fig: Figure | None = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> Axes:
        """Plot a window that helps in diagnosing basic model assumptions.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
            start time for which to calculate the residuals.
        tmax: str or pandas.Timestamp, optional
            end time for which to calculate the residuals.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        bins: int optional
            number of bins used for the histogram. 50 is default.
        acf_options: dict, optional
            dictionary with keyword arguments that are passed on to pastas.stats.acf.
        fig: matplotlib.pyplot.Figure, optional
            Optionally provide a matplotlib.pyplot.Figure instance to plot onto.
        alpha: float, optional
            Significance level to calculate the (1-alpha)-confidence intervals.
        **kwargs: dict, optional
            Optional keyword arguments, passed on to matplotlib.pyplot.figure method.

        Returns
        -------
        axes: list of matplotlib.axes.Axes

        Examples
        --------
        >>> axes = ml.plots.diagnostics()

        Notes
        -----
        This plot assumed that the noise or residuals follow a Normal distribution.

        See Also
        --------
        pastas.stats.acf
            Method that computes the autocorrelation.
        scipy.stats.probplot
            Method use to plot the probability plot.
        """
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax).iloc[1:]
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)

        sim = self.ml.simulate(tmin=tmin, tmax=tmax)

        if self.ml.interpolate_simulation:
            sim_interpolated = np.interp(res.index.asi8, sim.index.asi8, sim.values)
            sim = Series(index=res.index, data=sim_interpolated)

        return diagnostics(
            series=res,
            sim=sim,
            figsize=figsize,
            bins=bins,
            fig=fig,
            acf_options=acf_options,
            alpha=alpha,
            **kwargs,
        )

    @model_tmin_tmax
    def cum_frequency(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        ax: Axes | None = None,
        figsize: tuple = (5, 2),
        **kwargs,
    ) -> Axes:
        """Plot the cumulative frequency for the observations and simulation.

        Parameters
        ----------
        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        **kwargs:
            Passed on to plot_cum_frequency.

        Returns
        -------
        ax: matplotlib.axes.Axes

        See Also
        --------
        ps.stats.plot_cum_frequency
        """
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        return cum_frequency(obs, sim, ax=ax, figsize=figsize, **kwargs)

    def block_response(
        self,
        stressmodels: list[str] | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot the block response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        legend: bool, optional
            Boolean to determine to show the legend. Default is True.

        Returns
        -------
        matplotlib.axes.Axes
            matplotlib axes instance.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, **kwargs)

        if not stressmodels:
            stressmodels = self.ml.stressmodels.keys()

        legend = []

        for name in stressmodels:
            if hasattr(self.ml.stressmodels[name], "rfunc"):
                self.ml.get_block_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel %s not in stressmodels list.", name)

        ax.set_xlim(0)
        ax.set_xlabel("Time [days]")
        if legend:
            ax.legend(legend)
        return ax

    def step_response(
        self,
        stressmodels: list[str] | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot the step response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        legend: bool, optional
            Boolean to determine to show the legend. Default is True.

        Returns
        -------
        matplotlib.axes.Axes
            matplotlib axes instance.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, **kwargs)

        if not stressmodels:
            stressmodels = self.ml.stressmodels.keys()

        legend = []

        for name in stressmodels:
            if hasattr(self.ml.stressmodels[name], "rfunc"):
                self.ml.get_step_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel %s not in stressmodels list.", name)

        ax.set_xlim(0)
        ax.set_xlabel("Time [days]")
        if legend:
            ax.legend(legend)
        return ax

    @model_tmin_tmax
    def stresses(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        cols: int = 1,
        split: bool = True,
        sharex: bool = True,
        figsize: tuple = (10, 8),
        **kwargs,
    ) -> Axes:
        """This method creates a graph with all the stresses used in the model.

        Parameters
        ----------
        tmin: str or pd.Timestamp, optional
        tmax: str or pd.Timestamp, optional
        cols: int
            number of columns used for plotting.
        split: bool, optional
            Split the stress
        sharex: bool, optional
            Sharex the x-axis.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.

        Returns
        -------
        axes: matplotlib.axes.Axes
            matplotlib axes instance.
        """
        stresses = _get_stress_series(self.ml, split=split)

        rows = len(stresses)
        rows = -(-rows // cols)  # round up without additional import

        fig, axes = plt.subplots(rows, cols, sharex=sharex, figsize=figsize, **kwargs)

        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for ax, stress in zip(axes, stresses):
            stress.plot(ax=ax)
            ax.legend([stress.name], loc=2)

        plt.xlim(tmin, tmax)
        fig.tight_layout(pad=0.0)

        return axes

    @PastasDeprecationWarning(
        remove_version="1.6.0",
        reason=(
            "Quantifying contributions in one plot is ambiguous. "
            "Users are encouraged develop this themselves."
        ),
    )
    @model_tmin_tmax
    def contributions_pie(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        ax: Axes | None = None,
        figsize: Figure | None = None,
        split: bool = True,
        partition: str = "std",
        wedgeprops: dict | None = None,
        startangle: float = 90.0,
        autopct: str = "%1.1f%%",
        **kwargs,
    ) -> Axes:
        """Make a pie chart of the contributions. This plot is based on the TNO
        Groundwatertoolbox.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional.
        tmax: str or pandas.Timestamp, optional.
        ax: matplotlib.axes.Axes, optional
            The Axes to plot the pie chart on. A new figure and axes will be created of
            not provided.
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        partition : str
            statistic to use to determine contribution of stress, either 'sum' or
            'std' (default).
        wedgeprops: dict, optional, default None
            dict containing pie chart wedge properties, default is None, which sets
            edgecolor to white.
        startangle: float
            at which angle to start drawing wedges.
        autopct: str
            format string to add percentages to pie chart.
        kwargs: dict, optional
            The keyword arguments are passed on to plt.pie.

        Returns
        -------
        ax: matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        contribs = self.ml.get_contributions(split=split, tmin=tmin, tmax=tmax)
        if partition == "sum":
            # the part of each pie is determined by the sum of the contribution
            frac = [np.abs(contrib).sum() for contrib in contribs]
        elif partition == "std":
            # the part of each pie is determined by the std of the contribution
            frac = [contrib.std() for contrib in contribs]
        else:
            msg = "Unknown value for partition: {}".format(partition)
            raise (Exception(msg))

        # make sure the unexplained part is 100 - evp %
        evp = self.ml.stats.evp(tmin=tmin, tmax=tmax) / 100
        frac = np.array(frac) / sum(frac) * evp
        frac = np.append(frac, 1 - evp)

        if "labels" not in kwargs:
            labels = [contrib.name for contrib in contribs]
            labels.append("Unexplained")
            kwargs["labels"] = labels

        if wedgeprops is None:
            wedgeprops = {"edgecolor": "w"}

        ax.pie(
            frac,
            wedgeprops=wedgeprops,
            startangle=startangle,
            autopct=autopct,
            **kwargs,
        )
        ax.axis("equal")
        return ax

    @model_tmin_tmax
    def stacked_results(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        figsize: tuple = (10, 8),
        stackcolors: dict[str, str] | list[str] | None = None,
        stacklegend: bool = False,
        stacklegend_kws: dict | None = None,
        **kwargs,
    ) -> Axes:
        """Create a results plot, similar to `ml.plots.results()`, in which the
        individual contributions of stresses (in stressmodels with multiple stresses)
        are stacked.

        Parameters
        ----------
        tmin : str or pandas.Timestamp, optional
        tmax : str or pandas.Timestamp, optional
        figsize : tuple, optional
        stackcolors : dict or list, optional
            Either dictionary with stress names as keys and colors as values, or a
            list of colors. By default None which applies colors according to the
            order of stresses in the StressModel. Passing a dictionary can be useful
            to apply the same color to each stress across multiple figures.
        stacklegend : bool, optional
            Add legend to the stacked plot.
        stacklegend_kws : dict, optional
            dict with keyword arguments for stackplot legend


        Returns
        -------
        axes: list of axes objects
        """

        # Contribution per stress on model results plot
        def custom_sort(t):
            """Sort by mean contribution."""
            return t[1].mean()

        # Create standard results plot
        axes = self.ml.plots.results(tmin=tmin, tmax=tmax, figsize=figsize, **kwargs)

        nsm = len(self.ml.stressmodels)

        # loop over axes showing stressmodel contributions
        for i, sm in zip(range(3, 3 + 2 * nsm, 2), self.ml.stressmodels.keys()):
            # Get the contributions for StressModels with multiple stresses
            contributions = []
            sml = self.ml.stressmodels[sm]
            if (len(sml.stress) > 0) and (sml._name == "WellModel"):
                if stackcolors is None:
                    stackcolors = {
                        wnam: f"C{iw + 1}"
                        for iw, wnam in enumerate(sml.distances.index)
                    }
                elif isinstance(stackcolors, list):
                    stackcolors = {
                        name: icolor
                        for name, icolor in zip(sml.distances.index, stackcolors)
                    }
                elif not isinstance(stackcolors, dict):
                    raise TypeError("stackcolors must be None, list, or dict.")
                nsplit = sml.get_nsplit()
                ax_step = axes[i]  # step response axis
                ax_step.lines[0].remove()  # remove step response for r=1 m
                if nsplit > 1:
                    for istress in range(len(sml.stress)):
                        h = self.ml.get_contribution(
                            sm, istress=istress, tmin=tmin, tmax=tmax
                        )
                        name = sml.stress[istress].name
                        if name is None:
                            name = sm
                        contributions.append((name, h))

                        # plot step responses for each well, scaled with distance
                        p = sml.get_parameters(model=self.ml, istress=istress)
                        step = self.ml.get_step_response(sm, p=p)
                        ax_step.plot(step.index, step, c=stackcolors[name], label=name)
                        # recalculate y-limits step response axes
                        ax_step.relim()
                else:
                    h = self.ml.get_contribution(sm, tmin=tmin, tmax=tmax)
                    name = sm
                    contributions.append((name, h))

                contributions.sort(key=custom_sort)

                # add stacked plot to correct axes
                ax = axes[i - 1]
                ax.lines[0].remove()  # delete existing line

                names = [c[0] for c in contributions]  # get names
                contrib = [c[1] for c in contributions]  # get time series
                vstack = concat(contrib, axis=1, sort=False)
                colors = [stackcolors[name] for name in names]
                ax.stackplot(vstack.index, vstack.values.T, colors=colors, labels=names)
                if stacklegend:
                    if stacklegend_kws is None:
                        stacklegend_kws = {}
                    ncol = stacklegend_kws.pop("ncol", 5)
                    fontsize = stacklegend_kws.pop("fontsize", 6)
                    loc = stacklegend_kws.pop("loc", "best")

                    ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, **stacklegend_kws)

                # y-scale does not show 0
                ylower, yupper = ax.get_ylim()
                if (ylower < 0) and (yupper < 0):
                    ax.set_ylim(top=0)
                elif (ylower > 0) and (yupper > 0):
                    ax.set_ylim(bottom=0)

        return axes

    @model_tmin_tmax
    def series(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        split: bool = True,
        **kwargs,
    ) -> Axes:
        """Method to plot all the time series going into a Pastas Model.

        Parameters
        ----------
        tmin: str or pd.Timestamp
        tmax: str or pd.Timestamp
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        hist: bool
            Histogram for the Series. Returns the number of observations, mean,
            skew and kurtosis as well. For the head series the result of the
            shapiro-wilk test (p > 0.05) for normality is reported.
        bins: float
            Number of bins in the histogram plot.
        titles: bool
            Set the titles or not. Taken from the name attribute of the Series.
        labels: list of str
            List with the labels for each subplot.
        figsize: tuple
            Set the size of the figure.

        Returns
        -------
        matplotlib.axes.Axes
        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        stresses = _get_stress_series(self.ml, split=split)
        axes = series(obs, stresses=stresses, **kwargs)
        return axes

    @model_tmin_tmax
    def summary(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        results_kwargs: dict | None = None,
        diagnostics_kwargs: dict | None = None,
    ) -> Figure:
        """Create a plot with the results and diagnostics plot.

        Parameters
        ----------
        tmin: str or pd.Timestamp, optional
        tmax: str or pd.Timestamp, optional
        fname: str, optional
            string with the file name / path to store the PDF file.
        dpi: int, optional
            dpi to save the figure with.
        results_kwargs: dict, optional
            dictionary passed on to ml.plots.results method.
        diagnostics_kwargs: dict, optional
            dictionary passed on to ml.plots.diagnostics method.

        Returns
        -------
        fig: matplotlib.pyplot.Figure instance
        """

        if results_kwargs is None:
            results_kwargs = {}

        if diagnostics_kwargs is None:
            diagnostics_kwargs = {}

        fig = plt.figure(figsize=(8.27, 11.69), dpi=50)

        fig1, fig2 = fig.subfigures(2, 1, height_ratios=[1.25, 1.0])

        self.results(fig=fig1, tmin=tmin, tmax=tmax, **results_kwargs)
        self.diagnostics(fig=fig2, tmin=tmin, tmax=tmax, **diagnostics_kwargs)
        fig2.subplots_adjust(wspace=0.2)

        fig1.suptitle("Model Results", fontweight="bold")
        fig2.suptitle("Model Diagnostics", fontweight="bold")

        plt.subplots_adjust(left=0.1, top=0.9, right=0.95, bottom=0.1)
        return fig

    @model_tmin_tmax
    def summary_pdf(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        results_kwargs: dict | None = None,
        diagnostics_kwargs: dict | None = None,
        fname: str | None = None,
        dpi: int = 150,
    ) -> Figure:
        """Create a PDF file (A4) with the results and diagnostics plot.

        Parameters
        ----------
        tmin: str or pd.Timestamp, optional
        tmax: str or pd.Timestamp, optional
        results_kwargs: dict, optional
            dictionary passed on to ml.plots.results method.
        diagnostics_kwargs: dict, optional
            dictionary passed on to ml.plots.diagnostics method.
        fname: str, optional
            string with the file name / path to store the PDF file.
        dpi: int, optional
            dpi to save the figure with.

        Returns
        -------
        fig: matplotlib.pyplot.Figure instance
        """
        fname = "{}.pdf".format(self.ml.name) if fname is None else fname
        pdf = PdfPages(fname)
        fig = self.summary(
            tmin=tmin,
            tmax=tmax,
            results_kwargs=results_kwargs,
            diagnostics_kwargs=diagnostics_kwargs,
        )
        pdf.savefig(fig, orientation="portrait", dpi=dpi)
        pdf.close()
        return fig

    @model_tmin_tmax
    def pairplot(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        bins: int | None = None,
        split: bool = True,
    ) -> dict[str, Axes]:
        """Method to plot the correlation between all the time series going
        into a Pastas Model.

        Parameters
        ----------
        tmin: str or pd.Timestamp
        tmax: str or pd.Timestamp
        bins : int | None, optional
            Number of bins in the histogram, by default None which uses Sturge's
            rule to determine the number bins
        split: bool, optional
            Split the stresses in multiple stresses when possible.

        Returns
        -------
        matplotlib.axes.Axes
        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        stresses = _get_stress_series(self.ml, split=split)
        series = [obs] + list(stresses)
        axd = pairplot(data=series, bins=bins)
        return axd

    @model_tmin_tmax
    def contribution(
        self,
        tmin: TimestampType | None = None,
        tmax: TimestampType | None = None,
        name: str | None = None,
        plot_stress: bool = True,
        plot_response: bool = False,
        block_or_step: str = "step",
        istress: int | None = None,
        ax: Axes | None = None,
        **kwargs,
    ):
        """Plot the contribution of a stressmodel and optionally the stress and the response.

        Parameters
        ----------
        tmin: str or pd.Timestamp, optional
        tmax: str or pd.Timestamp, optional
        name: str, optional
            Name of the stressmodel to plot the contribution for.
        plot_stress: bool, optional
            Plot the stress on an overlay axes.
        plot_response: bool, optional
            Plot the step response on a separate axes on the right.
        block_or_step: str, optional
            Type of response to plot, either 'block' or 'step'. Default is 'step'.
        istress: int, optional
            Index of the stress to plot the response for. Default is None.
        ax: dict or matplotlib.axes.Axes, optional
            Dictionary containing axes with 'con' and 'rf' as keys, or a single axes
            instance for the contribution plot.
        kwargs: dict, optional
            Passed to the stress plot.

        Returns
        -------
        axes: dict
            Dictionary containing the axes for the contribution, and optionally the
            stress and response.
        """
        if name is None:
            raise ValueError(
                "Please provide a name for the stressmodel: "
                f"{list(self.ml.stressmodels.keys())}"
            )
        c = self.ml.get_contribution(name, tmin=tmin, tmax=tmax, istress=istress)

        if ax is None:
            if plot_response:
                _, axes = plt.subplot_mosaic(
                    [["con", "con", "con", "con", "rf"]],
                    constrained_layout=True,
                    figsize=(10, 2),
                )

            else:
                _, axes = plt.subplot_mosaic(
                    [["con"]],
                    constrained_layout=True,
                    figsize=(10, 2),
                )
        else:
            if not isinstance(ax, dict):
                axes = {"con": ax}

        axes["con"].plot(c.index, c, label=f"contribution {c.name}")

        if plot_stress:
            sm = self.ml.stressmodels[name]
            # get stress
            if sm._name == "RechargeModel":
                # compute recharge
                s = sm.get_stress(tmin=tmin, tmax=tmax, istress=istress)
                stress_name = s.name
            else:
                s = self.ml.get_stress(name, tmin=tmin, tmax=tmax, istress=istress)
                # if multiple stresses, sum stresses together
                if isinstance(s, list):
                    s = concat(s, axis=1).sum(axis=1, fill_value=0.0)
                    stress_name = name
                else:
                    stress_name = s.name

            # use up to flip stress if necessary
            up = 1.0 if sm.rfunc.up in [True, None] else -1.0

            # add second axes for stress
            axes["stress"] = axes["con"].twinx()
            if "c" not in kwargs:
                color = kwargs.pop("color", (0.4, 0.4, 0.4))
            axes["stress"].plot(
                s.index,
                up * s,
                color=color,
                lw=1.0,
                label="stress",
                **kwargs,
            )
            axes["stress"].set_ylabel(f"stress '{stress_name}'")
            # flip order of stress and contributions axes (contributions on top)
            axes["con"].patch.set_visible(False)
            axes["stress"].patch.set_visible(True)
            axes["con"].set_zorder(axes["stress"].get_zorder() + 1)
            # add both lines to legend
            h1, l1 = axes["con"].get_legend_handles_labels()
            h2, l2 = axes["stress"].get_legend_handles_labels()
            axes["con"].legend(
                h1 + h2, l1 + l2, loc=(0, 1), frameon=False, ncol=2, fontsize="small"
            )
        else:
            axes["con"].legend(loc=(0, 1), frameon=False, ncol=1, fontsize="small")

        if plot_response:
            if "rf" not in axes:
                raise ValueError(
                    "No axes defined for response. "
                    "Provide a dictionary containing axes with 'con' and 'rf' as keys."
                )
            if block_or_step == "step":
                self.step_response(stressmodels=[name], ax=axes["rf"], legend=False)
            else:
                self.block_response(stressmodels=[name], ax=axes["rf"], legend=False)
            axes["rf"].yaxis.set_label_position("right")
            axes["rf"].yaxis.tick_right()
            h3, _ = axes["rf"].get_legend_handles_labels()
            if len(h3) == 1:
                axes["rf"].legend(
                    h3,
                    [f"{block_or_step} response"],
                    loc=(0, 1),
                    frameon=False,
                    fontsize="small",
                )
            axes["rf"].grid(True)

        axes["con"].grid(True)
        axes["con"].set_ylabel("rise")
        return axes
