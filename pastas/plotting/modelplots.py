"""Plotting methods for Pastas Models, including time series and diagnostics plots."""

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import LogFormatter, MultipleLocator
from pandas import DataFrame, Series, Timestamp, concat

from pastas.decorators import (
    PastasDeprecationWarning,
    deprecate_args_or_kwargs,
    model_tmin_tmax,
)
from pastas.plotting.plots import cum_frequency, diagnostics, pairplot, series
from pastas.plotting.plotutil import (
    _get_height_ratios,
    _get_stress_series,
    _table_formatter_params,
    _table_formatter_stderr,
    plot_series_with_gaps,
    share_xaxes,
)
from pastas.timeseries_utils import _index_to_int64
from pastas.typing import Axes, Figure, Model, StressModel

logger = logging.getLogger(__name__)


class Plotting:
    """Class that contains all plotting methods for Pastas models.

    Pastas models come with a number of predefined plotting methods to quickly
    visualize a Model. All of these methods are contained in the `plot` attribute of
    a model. For example, if we stored a :class:`pastas.model.Model` instance in the
    variable `ml`, the plot methods are available as follows::

        ml.plots.results()

    """

    def __init__(self, ml: Model) -> None:
        self.ml = ml  # Store a reference to the model class

    def __repr__(self) -> str:
        """Return a string representation of the ModelPlots class."""
        msg = (
            "This module contains all the built-in plotting options that are available."
        )
        return msg

    @model_tmin_tmax
    def plot(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        oseries: bool = True,
        simulation: bool = True,
        ax: Axes | None = None,
        legend: bool = True,
        **kwargs,
    ) -> Axes:
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        oseries: bool, optional
            True to plot the observed time series.
        simulation: bool, optional
            True to plot the simulated time series.
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
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
        kwargs = {} or kwargs
        if ax is None:
            layout = kwargs.pop("layout", "tight")
            figsize = kwargs.pop("figsize", (8.0, 4.0))
            _, ax = plt.subplots(figsize=figsize, layout=layout, **kwargs)

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o_nu = self.ml.oseries.series_original.drop(o.index).loc[
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

        return ax

    @model_tmin_tmax
    def results(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        split_contributions: bool = False,
        all_responses: bool = False,
        adjust_height: bool = True,
        return_warmup: bool = False,
        add_ylabels: bool = True,
        block_or_step: Literal["block", "step"] = "step",
        stderr: bool = False,
        return_dict: bool = False,
        **kwargs,
    ) -> dict[str, Axes] | list[Axes]:
        """Plot the results of the model in a mosaic plot.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        split_contributions: bool, optional
            Split the contributions in multiple stresses when possible. Default is
            False.
        all_responses: bool, optional
            Plot all responses if True. If False, only the first response per
            contribution is plotted. Default is False.
        adjust_height: bool, optional
            Adjust the height of the graphs, so that the vertical scale of all the
            subplots on the left is equal. Default is True.
        return_warmup: bool, optional
            Show, not return, the warmup-period. Default is False.
        add_ylabels: bool, optional
            Add ylabels to the subplots. Default is True.
        block_or_step: {"block", "step"}, optional
            Plot the block- or step-response on the right. Default is 'step'.
        stderr : bool, optional
            If True the standard error of the parameter values are shown.
        return_dict: bool, optional
            If True, a dictionary with the axes is returned. If False, a list of
            axes is returned. Default is False.
        **kwargs: dict, optional
            Optional arguments, passed on to the matplotlib.pyplot.figure method.

        Returns
        -------
        Dictionary with the matplotlib.axes.Axes

        Examples
        --------
        >>> ml.plots.results_mosaic()
        """
        if "split" in kwargs:
            deprecate_args_or_kwargs(
                name="split",
                version="2.2.0",
                reason="Use `split_contributions` instead.",
            )
            split_contributions = kwargs.pop("split")

        tmin = Timestamp(tmin) if tmin is not None else None
        tmax = Timestamp(tmax) if tmax is not None else None

        # get simulated time series
        o = self.ml.observations(tmin=tmin, tmax=tmax)
        o_nu = self.ml.oseries.series_original.drop(o.index)
        o_nu = (
            o_nu[tmin - self.ml.settings["warmup"] : tmax]
            if return_warmup
            else o_nu[tmin:tmax]
        )
        sim = self.ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        contrib_list = self.ml.get_contributions(
            split=split_contributions,
            tmin=tmin,
            tmax=tmax,
            return_warmup=return_warmup,
        )

        contribs = {}
        rows = []
        i = 0
        for sm_name, sm in self.ml.stressmodels.items():
            nsplit = sm.nsplit if split_contributions else 1
            for istress in range(nsplit):
                suffix = sm_name if not split_contributions else f"{sm_name}_{istress}"
                con_key = f"con_{suffix}"
                rf_key = f"rf_{suffix}"
                contribs[con_key] = contrib_list[i]
                rows.append(
                    (con_key, rf_key, sm_name, istress if split_contributions else None)
                )
                i += 1

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

        for con_key, contrib in contribs.items():
            hs = contrib.loc[tmin:tmax]
            if hs.empty:
                if contrib.empty:
                    ylim_c = [0.0, 0.0]
                else:
                    ylim_c = [contrib.min(), hs.max()]
            else:
                ylim_c = [hs.min(), hs.max()]
            ylims[con_key] = ylim_c

        # construct mosoaic
        mosaic = [[x] for x in ylims]
        for mos in mosaic:
            if "con_" in mos[0]:
                mos.append(f"rf_{mos[0].split('_', 1)[1]}")
            elif mos[0] in ("sim", "res"):
                mos.append("tab")

        kwargs = {} or kwargs
        width_ratios = kwargs.pop("width_ratios", [2.0, 1.0])
        height_ratios = (
            _get_height_ratios(list(ylims.values()))
            if adjust_height
            else kwargs.pop("height_ratios", None)
        )
        figsize = kwargs.pop("figsize", (8.0, 4.0 + 2 * len(contribs)))
        layout = kwargs.pop("layout", "constrained")
        fig = kwargs.pop("fig", None)
        if fig is None:
            fig, axd = plt.subplot_mosaic(
                mosaic=mosaic,
                figsize=figsize,
                layout=layout,
                height_ratios=height_ratios,
                width_ratios=width_ratios,
                **kwargs,
            )
        else:
            axd = fig.subplot_mosaic(
                mosaic=mosaic,
                height_ratios=height_ratios,
                width_ratios=width_ratios,
                **kwargs,
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
        if self.ml.noisemodel is not None:
            noise = self.ml.noise(tmin=tmin, tmax=tmax)
            _ = plot_series_with_gaps(noise, ax=axd["res"], color="C0")
        axd["res"].axhline(0.0, color="k", linestyle="--", zorder=0)
        axd["res"].legend(loc=(0, 1), ncol=2, frameon=False)

        # plot the contributions and responses of the stressmodels
        for con_key, rf_key, sm_name, istress in rows:
            sm = self.ml.stressmodels[sm_name]
            axd[con_key].plot(
                contribs[con_key].index,
                contribs[con_key].values,
                label=contribs[con_key].name,
            )
            if not split_contributions:
                title = [stress.name for stress in sm.stresses]
                if len(title) > 3:
                    title = title[:3] + ["..."]
                if title:
                    axd[con_key].set_title(
                        "Stresses: " + str(title).replace("'", ""),
                        loc="right",
                        fontsize=plt.rcParams["legend.fontsize"],
                    )
            axd[con_key].legend(loc=(0, 1), ncol=1, frameon=False)
            axd[con_key].set_ylim(ylims[con_key])
            _ = self._plot_response_in_results(
                sm=sm,
                block_or_step=block_or_step,
                ax=axd[rf_key],
                istress=(
                    istress if split_contributions else (None if all_responses else 0)
                ),
            )

        # share x-axes of simulation, residuals and contributions
        share_xaxes([axd[k] for k in [x[0] for x in mosaic]])
        if return_warmup:
            axd["sim"].set_xlim(tmin - self.ml.settings["warmup"], tmax)
        else:
            axd["sim"].set_xlim(tmin, tmax)

        # add legend to the upper response axes and share x-axes of responses
        response_axes = [axd[k] for k in [x[1] for x in mosaic] if k.startswith("rf_")]
        response_axes[0].legend(loc=(0, 1), ncol=2, frameon=False)

        response_xlims = [ax.get_xlim() for ax in response_axes]
        share_xaxes(response_axes)
        response_axes[-1].set_xlim(
            left=min(x[0] for x in response_xlims),
            right=max(x[1] for x in response_xlims),
        )

        for k in axd:
            axd[k].grid(True)
            if k.startswith("rf_"):
                axd[k].yaxis.tick_right()
                axd[k].yaxis.set_label_position("right")
            if add_ylabels:
                if k == "sim":
                    axd[k].set_ylabel("Head")
                elif k == "res":
                    axd[k].set_ylabel("Error")
                elif k.startswith("con_"):
                    axd[k].set_ylabel("Rise")
                elif k.startswith("rf_"):
                    axd[k].set_ylabel("[unit head]/[unit stress]")

        _ = self._plot_parameters_table(ax=axd["tab"], stderr=stderr)

        fig.align_ylabels()

        return axd if return_dict else list(axd.values())

    @PastasDeprecationWarning(
        version="2.2.0", reason="Use `results` instead with the return_dict argument."
    )
    def results_mosaic(self, *args, **kwargs) -> dict[str, Axes]:
        """Plot the results of the model in a mosaic plot (deprecated).

        Deprecated: Use `results` instead with the return_dict argument to specify the layout
        of the mosaic plot.
        """
        kwargs = {} or kwargs
        kwargs["return_dict"] = True
        return self.results(*args, **kwargs)

    def _plot_response_in_results(
        self,
        sm: StressModel,
        block_or_step: Literal["step", "block"],
        ax: Axes,
        istress: int | None = None,
    ):
        """Plot the response of a Stressmodel in the results-plot."""
        responses = sm._get_responses(
            self.ml, block_or_step=block_or_step, istress=istress
        )
        responses = [x for x in responses if x is not None]
        if responses:
            # Keep the first cycle color for a single response, but reserve it
            # when plotting multiple responses.
            if len(responses) > 1:
                ax._get_lines.get_next_color()

            xlim_left = min(
                [
                    x.index[0] if block_or_step == "step" else x.index[1]
                    for x in responses
                    if x is not None
                ]
            )
            xlim_right = max([x.index[-1] for x in responses])
            for i, response in enumerate(responses):
                if i == 0 and block_or_step == "block":
                    ax.set_xscale("log")
                    ax.xaxis.set_major_formatter(LogFormatter())

                if len(responses) == 1:
                    label = f"{block_or_step.capitalize()} response"
                else:
                    label = response.name
                ax.plot(
                    response.index,
                    response.values,
                    label=label,
                    color=ax._get_lines.get_next_color(),
                )
                ax.set_xlim(left=xlim_left, right=xlim_right)
        return ax

    def _plot_parameters_table(self, ax: Axes, stderr: bool) -> None:
        """Plot the parameters table in the results-plot."""
        ax.set_title(
            f"Model parameters ($N_c$={self.ml.parameters.vary.sum()})",
            loc="left",
            fontsize=plt.rcParams["legend.fontsize"],
        )
        p = self.ml.parameters.loc[:, ["name"]].copy()
        p.loc[:, "name"] = p.index

        if self.ml.parameters.loc[:, "optimal"].isna().all():
            colnam = "initial"
        else:
            colnam = "optimal"

        p.loc[:, colnam] = self.ml.parameters.loc[:, colnam].apply(
            _table_formatter_params
        )
        if stderr:
            if "stderr" not in self.ml.parameters.columns:
                logger.error(
                    "Standard errors are not available in the model parameters."
                )
            else:
                stderrper = (
                    self.ml.parameters.loc[:, "stderr"]
                    / self.ml.parameters.loc[:, "optimal"]
                )
                p.loc[:, "stderr"] = stderrper.abs().apply(_table_formatter_stderr)
        ax.axis("off")
        raw_widths = [max(p[col].str.len().max(), len(col)) for col in p.columns]
        total = sum(raw_widths)
        col_widths = [w / total for w in raw_widths]
        ax.table(
            bbox=(0.0, 0.0, 1.0, 1.0),
            cellText=p.values,
            colWidths=col_widths,
            colLabels=p.columns,
        )
        return ax

    @model_tmin_tmax
    def decomposition(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        ytick_base: bool = True,
        split_contributions: bool = True,
        axes: Axes | None = None,
        name: str | None = None,
        return_warmup: bool = False,
        min_ylim_diff: float | None = None,
        **kwargs,
    ) -> list[Axes]:
        """Plot the decomposition of a time-series in the different stresses.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        ytick_base: Boolean or float, optional
            Make the ytick-base constant if True, set this base to float if a float.
        split_contributions: bool, optional
            Split the stresses in multiple stresses when possible. Default is True.
        axes: matplotlib.axes.Axes instance, optional
            Matplotlib Axes instance to plot the figure on to.
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
        kwargs = {} or kwargs
        if "split" in kwargs:
            deprecate_args_or_kwargs(
                name="split",
                version="2.3.0",
                reason="Use `split_contributions` instead.",
            )
            split_contributions = kwargs.pop("split")

        o = self.ml.observations(tmin=tmin, tmax=tmax)

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
        if name is not None:
            sim.name = name

        # determine the influence of the different stresses
        contribs = self.ml.get_contributions(
            split=split_contributions,
            tmin=tmin,
            tmax=tmax,
            return_warmup=return_warmup,
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
            layout = kwargs.pop("layout", "tight")
            figsize = kwargs.pop("figsize", (8.0, 2.0 + 1.5 * len(contribs)))
            fig, axes = plt.subplots(
                nrows=nrows,
                sharex=True,
                figsize=figsize,
                gridspec_kw=gridspec_kw,
                layout=layout,
                **kwargs,
            )
            axes = np.atleast_1d(axes)
            o_label = o.name
            set_axes_properties = True
        else:
            if len(axes) != nrows:
                msg = "Makes sure the number of axes equals the number of series"
                raise ValueError(msg)
            fig = axes[0].figure
            o_label = ""
            set_axes_properties = False

        # plot simulation and observations in top graph
        o_nu = self.ml.oseries.series_original.drop(o.index)
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

        return fig.axes

    @model_tmin_tmax
    def diagnostics(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        bins: int = 50,
        acf_options: dict | None = None,
        fig: Figure | None = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> Axes:
        """Plot a window that helps in diagnosing basic model assumptions.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            start time for which to calculate the residuals.
        tmax: pandas.Timestamp or str, optional
            end time for which to calculate the residuals.
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
        if self.ml.noisemodel is not None:
            res = self.ml.noise(tmin=tmin, tmax=tmax).iloc[1:]
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)

        sim = self.ml.simulate(tmin=tmin, tmax=tmax)

        if self.ml._interpolate_simulation:
            sim_interpolated = np.interp(
                _index_to_int64(res.index),
                _index_to_int64(sim.index),
                sim.values,
            )
            sim = Series(index=res.index, data=sim_interpolated)

        return diagnostics(
            series=res,
            sim=sim,
            bins=bins,
            fig=fig,
            acf_options=acf_options,
            alpha=alpha,
            **kwargs,
        )

    @model_tmin_tmax
    def cum_frequency(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> Axes:
        """Plot the cumulative frequency for the observations and simulation.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        ax: matplotlib.axes.Axes, optional
            Axes to add the plot to.
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
        return cum_frequency(obs=obs, sim=sim, ax=ax, **kwargs)

    def block_response(
        self,
        stressmodels: list[str] | None = None,
        ax: Axes | None = None,
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
        kwargs = {} or kwargs
        if ax is None:
            figsize = kwargs.pop("figsize", (5.0, 3.0))
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
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        cols: int = 1,
        split: bool = True,
        sharex: bool = True,
        figsize: tuple = (10, 8),
        **kwargs,
    ) -> list[Axes]:
        """Create a graph with all the stresses used in the model.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
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
        axes: list[matplotlib.axes.Axes]
            List of matplotlib axes instances.
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
            ax.set_xlim(tmin, tmax)

        return fig.axes

    @PastasDeprecationWarning(
        version="1.6.0",
        reason=(
            "Quantifying contributions in one plot is ambiguous. "
            "Users are encouraged develop this themselves."
        ),
    )
    @model_tmin_tmax
    def contributions_pie(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        ax: Axes | None = None,
        figsize: Figure | None = None,
        split: bool = True,
        partition: str = "std",
        wedgeprops: dict | None = None,
        startangle: float = 90.0,
        autopct: str = "%1.1f%%",
        **kwargs,
    ) -> Axes:
        """Make a pie chart of the contributions.

        This plot is based on the TNO Groundwatertoolbox.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
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
            raise ValueError(msg)

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
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        stackcolors: dict[str, str] | list[str] | None = None,
        stacklegend: bool = False,
        stacklegend_kws: dict | None = None,
        **kwargs,
    ) -> list[Axes]:
        """Create a results plot, similar to `ml.plots.results()`.

        In this plot, the individual contributions of stresses (in stressmodels with
        multiple stresses) are stacked.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
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
        # Create standard results plot
        kwargs["return_dict"] = True
        axd = self.ml.plots.results(tmin=tmin, tmax=tmax, **kwargs)
        # loop over axes showing stressmodel contributions
        for sm_name, sm in self.ml.stressmodels.items():
            # Get the contributions for StressModels with multiple stresses
            contributions = {}
            if sm.stresses and (sm._name == "WellModel"):
                if stackcolors is None:
                    stackcolors = {
                        wnam: f"C{i + 1}" for i, wnam in enumerate(sm.stresses._fields)
                    }
                    stackcolors[sm_name] = (
                        "C0"  # add backup for single-stress WellModels
                    )
                elif isinstance(stackcolors, (list, tuple)):
                    stackcolors = dict(zip(sm.stresses._fields, stackcolors))
                elif not isinstance(stackcolors, dict):
                    raise TypeError("stackcolors must be None, list, or dict.")
                if sm.nsplit > 1:
                    axd[f"rf_{sm_name}"].lines[
                        0
                    ].remove()  # remove step response for r=1 m
                    for istress in range(len(sm.stresses)):
                        h = self.ml.get_contribution(
                            sm_name, istress=istress, tmin=tmin, tmax=tmax
                        )
                        name = sm.stresses[istress].name
                        name = sm if name is None else name
                        contributions[name] = h

                        # plot step responses for each well, scaled with distance
                        p = sm.get_parameters(model=self.ml, istress=istress)
                        step = self.ml.get_step_response(sm_name, p=p)
                        axd[f"rf_{sm_name}"].plot(
                            step.index, step, c=stackcolors[name], label=name
                        )
                        axd[f"rf_{sm_name}"].relim()
                else:
                    contributions[sm_name] = self.ml.get_contribution(
                        sm_name, tmin=tmin, tmax=tmax
                    )
                contributions_df = concat(contributions, axis=1, sort=False)
                order = contributions_df.mean(axis=0).sort_values(ascending=False).index
                contributions_df = contributions_df[order]

                # add stacked plot to correct axes
                axd[f"con_{sm_name}"].lines[0].remove()  # delete existing line

                colors = [stackcolors[name] for name in contributions_df.columns]
                axd[f"con_{sm_name}"].stackplot(
                    contributions_df.index,
                    contributions_df.values.T,
                    colors=colors,
                    labels=contributions_df.columns,
                )
                if stacklegend:
                    if stacklegend_kws is None:
                        stacklegend_kws = {}
                    ncol = stacklegend_kws.pop("ncol", 5)
                    fontsize = stacklegend_kws.pop("fontsize", 6)
                    loc = stacklegend_kws.pop("loc", "best")

                    axd[f"con_{sm_name}"].legend(
                        loc=loc, ncol=ncol, fontsize=fontsize, **stacklegend_kws
                    )

                # y-scale does not show 0
                ylower, yupper = axd[f"con_{sm_name}"].get_ylim()
                if (ylower < 0) and (yupper < 0):
                    axd[f"con_{sm_name}"].set_ylim(top=0)
                elif (ylower > 0) and (yupper > 0):
                    axd[f"con_{sm_name}"].set_ylim(bottom=0)

        return list(axd.values())

    @model_tmin_tmax
    def series(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        split: bool = True,
        **kwargs,
    ) -> Axes:
        """Plot all the time series going into a Pastas Model.

        Parameters
        ----------
        tmin: str or Timestamp
        tmax: str or Timestamp
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
        ax = series(obs, stresses=stresses, **kwargs)
        return ax

    @model_tmin_tmax
    def summary(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        results_kwargs: dict | None = None,
        diagnostics_kwargs: dict | None = None,
    ) -> Figure:
        """Create a plot with the results and diagnostics plot.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
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
        fig = plt.figure(figsize=(8.27, 11.69), dpi=50, layout="constrained")
        fig1, fig2 = fig.subfigures(2, 1, height_ratios=[2, 1], hspace=0.08)

        self.results(fig=fig1, tmin=tmin, tmax=tmax, **results_kwargs or {})
        self.diagnostics(fig=fig2, tmin=tmin, tmax=tmax, **diagnostics_kwargs or {})

        fig1.suptitle("Model Results", fontweight="bold")
        fig2.suptitle("Model Diagnostics", fontweight="bold")

        return fig

    @model_tmin_tmax
    def summary_pdf(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        results_kwargs: dict | None = None,
        diagnostics_kwargs: dict | None = None,
        fname: str | None = None,
        dpi: int = 150,
    ) -> Figure:
        """Create a PDF file (A4) with the results and diagnostics plot.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
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
        fig = self.summary(
            tmin=tmin,
            tmax=tmax,
            results_kwargs=results_kwargs,
            diagnostics_kwargs=diagnostics_kwargs,
        )
        with PdfPages(fname) as pdf:
            pdf.savefig(fig, orientation="portrait", dpi=dpi)
        return fig

    @model_tmin_tmax
    def pairplot(
        self,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        bins: int | None = None,
        split: bool = True,
    ) -> dict[str, Axes]:
        """Plot the correlation between all the time series going into a Pastas Model.

        Parameters
        ----------
        tmin: str or Timestamp
        tmax: str or Timestamp
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
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        name: str | None = None,
        plot_stress: bool = True,
        plot_response: bool = False,
        block_or_step: Literal["block", "step"] = "step",
        istress: int | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> dict[str, Axes]:
        """Plot the contribution of a stressmodel and optionally the stress and the response.

        Parameters
        ----------
        tmin: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the start date for the period
            (E.g. '1980-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        tmax: pandas.Timestamp or str, optional
            A string or pandas.Timestamp with the end date for the period
            (E.g. '2020-01-01 00:00:00'). Strings are converted to
            pandas.Timestamp internally.
        name: str, optional
            Name of the stressmodel to plot the contribution for.
        plot_stress: bool, optional
            Plot the stress on an overlay axes.
        plot_response: bool, optional
            Plot the step response on a separate axes on the right.
        block_or_step: {"block", "step"}, optional
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
                _, axd = plt.subplot_mosaic(
                    [["con", "rf"]],
                    width_ratios=[4, 1],
                    constrained_layout=True,
                    figsize=(8.0, 2.0),
                )

            else:
                _, axd = plt.subplot_mosaic(
                    [["con"]],
                    constrained_layout=True,
                    figsize=(8.0, 2.0),
                )
        else:
            if not isinstance(ax, dict):
                axd = {"con": ax}
            else:
                axd = ax

        axd["con"].plot(c.index, c, label=f"contribution {c.name}")

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
                    s = concat(s, axis=1).sum(axis=1, skipna=True)
                    stress_name = name
                elif isinstance(s, DataFrame):
                    s = s.sum("columns", skipna=True)
                    stress_name = name
                else:
                    stress_name = s.name

            # use up to flip stress if necessary
            up = 1.0 if sm.rfunc.up in [True, None] else -1.0

            # add second axes for stress
            axd["stress"] = axd["con"].twinx()
            if "c" not in kwargs:
                color = kwargs.pop("color", (0.4, 0.4, 0.4))
            axd["stress"].plot(
                s.index,
                up * s,
                color=color,
                lw=1.0,
                label="stress",
                **kwargs,
            )
            axd["stress"].set_ylabel(f"stress '{stress_name}'")
            # flip order of stress and contributions axes (contributions on top)
            axd["con"].patch.set_visible(False)
            axd["stress"].patch.set_visible(True)
            axd["con"].set_zorder(axd["stress"].get_zorder() + 1)
            # add both lines to legend
            h1, l1 = axd["con"].get_legend_handles_labels()
            h2, l2 = axd["stress"].get_legend_handles_labels()
            axd["con"].legend(
                h1 + h2, l1 + l2, loc=(0, 1), frameon=False, ncol=2, fontsize="small"
            )
        else:
            axd["con"].legend(loc=(0, 1), frameon=False, ncol=1, fontsize="small")

        if plot_response:
            if "rf" not in axd:
                raise ValueError(
                    "No axes defined for response. "
                    "Provide a dictionary containing axes with 'con' and 'rf' as keys."
                )
            if block_or_step == "step":
                self.step_response(stressmodels=[name], ax=axd["rf"], legend=False)
            else:
                self.block_response(stressmodels=[name], ax=axd["rf"], legend=False)
            axd["rf"].yaxis.set_label_position("right")
            axd["rf"].yaxis.tick_right()
            h3, _ = axd["rf"].get_legend_handles_labels()
            if len(h3) == 1:
                axd["rf"].legend(
                    h3,
                    [f"{block_or_step} response"],
                    loc=(0, 1),
                    frameon=False,
                    fontsize="small",
                )
            axd["rf"].grid(True)

        axd["con"].grid(True)
        axd["con"].set_ylabel("rise")
        return axd
