"""This module contains all the plotting methods for Pastas Models.

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, LogFormatter
from pandas import concat

from .decorators import model_tmin_tmax
from .plots import series, diagnostics, cum_frequency, \
    _table_formatter_params, _table_formatter_stderr

logger = logging.getLogger(__name__)


class Plotting:
    """Class that contains all plotting methods for Pastas models.

    Pastas models come with a number of predefined plotting methods to quickly
    visualize a Model. All of these methods are contained in the `plot`
    attribute of a model. For example, if we stored a
    :class:`pastas.model.Model` instance in the variable `ml`, the plot
    methods are available as follows::

    >>> ml.plot.results()

    """

    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that " \
              "are available."
        return msg

    @model_tmin_tmax
    def plot(self, tmin=None, tmax=None, oseries=True, simulation=True,
             ax=None, figsize=None, legend=True, **kwargs):
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        oseries: bool, optional
            True to plot the observed time series.
        simulation: bool, optional
            True to plot the simulated time series.
        ax: Matplotlib.axes instance, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        legend: bool, optional
            Boolean to determine to show the legend (True) or not (False).

        Returns
        -------
        ax: matplotlib.axes.Axes
            matplotlib axes with the simulated and optionally the observed
            timeseries.

        Examples
        --------
        >>> ml.plot()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, **kwargs)

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o_nu = self.ml.oseries.series.drop(o.index).loc[
                o.index.min():o.index.max()]
            if not o_nu.empty:
                # plot parts of the oseries that are not used in grey
                o_nu.plot(linestyle='', marker='.', color='0.5', label='',
                          ax=ax)
            o.plot(linestyle='', marker='.', color='k', ax=ax)

        if simulation:
            sim = self.ml.simulate(tmin=tmin, tmax=tmax)
            r2 = round(self.ml.stats.rsq(tmin=tmin, tmax=tmax) * 100, 1)
            sim.plot(ax=ax, label=f'{sim.name} ($R^2$ = {r2}%)')

        # Dress up the plot
        ax.set_xlim(tmin, tmax)
        ax.set_ylabel("Groundwater levels [meter]")
        ax.set_title("Results of {}".format(self.ml.name))

        if legend:
            ax.legend(ncol=2, numpoints=3)
        plt.tight_layout()
        return ax

    @model_tmin_tmax
    def results(self, tmin=None, tmax=None, figsize=(10, 8), split=False,
                adjust_height=True, return_warmup=False, block_or_step='step',
                fig=None, **kwargs):
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        split: bool, optional
            Split the stresses in multiple stresses when possible. Default is
            False.
        adjust_height: bool, optional
            Adjust the height of the graphs, so that the vertical scale of all
            the subplots on the left is equal. Default is True.
        return_warmup: bool, optional
            Show the warmup-period. Default is false.
        block_or_step: str, optional
            Plot the block- or step-response on the right. Default is 'step'.
        fig: Matplotib.Figure instance, optional
            Optionally provide a Matplotib.Figure instance to plot onto.
        **kwargs: dict, optional
            Optional arguments, passed on to the plt.figure method.

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
            o_nu = o_nu[tmin - self.ml.settings['warmup']: tmax]
        else:
            o_nu = o_nu[tmin: tmax]
        sim = self.ml.simulate(tmin=tmin, tmax=tmax,
                               return_warmup=return_warmup)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        contribs = self.ml.get_contributions(split=split, tmin=tmin,
                                             tmax=tmax,
                                             return_warmup=return_warmup)

        ylims = [(min([sim.min(), o[tmin:tmax].min()]),
                  max([sim.max(), o[tmin:tmax].max()])),
                 (res.min(), res.max())]  # residuals are bigger than noise

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

        gs = fig.add_gridspec(ncols=2, nrows=len(contribs) + 2,
                              width_ratios=[2, 1], height_ratios=hrs)

        # Main frame
        ax1 = fig.add_subplot(gs[0, 0])
        o.plot(ax=ax1, linestyle='', marker='.', color='k', x_compat=True)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(ax=ax1, linestyle='', marker='.', color='0.5', label='',
                      x_compat=True, zorder=-1)

        # add rsq to simulation
        r2 = self.ml.stats.rsq(tmin=tmin, tmax=tmax)
        sim.plot(ax=ax1, x_compat=True, label=f'{sim.name} ($R^2$={r2:.2%})')
        ax1.legend(loc=(0, 1), ncol=3, frameon=False, numpoints=3)
        ax1.set_ylim(ylims[0])

        # Residuals and noise
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        res.plot(ax=ax2, color='k', x_compat=True)
        if self.ml.settings["noise"] and self.ml.noisemodel:
            noise = self.ml.noise(tmin=tmin, tmax=tmax)
            noise.plot(ax=ax2, x_compat=True)
        ax2.axhline(0.0, color='k', linestyle='--', zorder=0)
        ax2.legend(loc=(0, 1), ncol=3, frameon=False)

        # Add a row for each stressmodel
        rmin = 0  # tmin of the response
        rmax = 0  # tmax of the response
        axb = None
        i = 0
        for sm_name, sm in self.ml.stressmodels.items():
            # plot the contribution
            nsplit = sm.get_nsplit()
            if split and nsplit > 1:
                for _ in range(nsplit):
                    ax = fig.add_subplot(gs[i + 2, 0], sharex=ax1)
                    contribs[i].plot(ax=ax, x_compat=True)
                    ax.legend(loc=(0, 1), ncol=3, frameon=False)
                    if adjust_height:
                        ax.set_ylim(ylims[i + 2])
                    i = i + 1
            else:
                ax = fig.add_subplot(gs[i + 2, 0], sharex=ax1)
                contribs[i].plot(ax=ax, x_compat=True)
                title = [stress.name for stress in sm.stress]
                if len(title) > 3:
                    title = title[:3] + ["..."]
                ax.set_title(f"Stresses: {title}", loc="right",
                             fontsize=plt.rcParams['legend.fontsize'])
                ax.legend(loc=(0, 1), ncol=3, frameon=False)
                if adjust_height:
                    ax.set_ylim(ylims[i + 2])
                i = i + 1

            # plot the step response
            response = self.ml._get_response(block_or_step=block_or_step,
                                             name=sm_name, add_0=True)

            if response is not None:
                rmax = max(rmax, response.index.max())
                axb = fig.add_subplot(gs[i + 1, 1], sharex=axb)
                response.plot(ax=axb)
                if block_or_step == 'block':
                    title = 'Block response'
                    rmin = response.index[1]
                    axb.set_xscale('log')
                    axb.xaxis.set_major_formatter(LogFormatter())
                else:
                    title = 'Step response'
                axb.set_title(title, fontsize=plt.rcParams['legend.fontsize'])

        if axb is not None:
            axb.set_xlim(rmin, rmax)

        # xlim sets minorticks back after plots:
        ax1.minorticks_off()

        if return_warmup:
            ax1.set_xlim(tmin - self.ml.settings['warmup'], tmax)
        else:
            ax1.set_xlim(tmin, tmax)

        # sometimes, ticks suddenly appear on top plot, turn off just in case
        plt.setp(ax1.get_xticklabels(), visible=False)

        for ax in fig.axes:
            ax.grid(True)

        if isinstance(fig, plt.Figure):
            fig.tight_layout(pad=0.0)  # Before making the table

        # Draw parameters table
        ax3 = fig.add_subplot(gs[0:2, 1])
        n_free = self.ml.parameters.vary.sum()
        ax3.set_title(f'Model Parameters ($n_c$={n_free})', loc='left',
                      fontsize=plt.rcParams['legend.fontsize'])
        p = self.ml.parameters.copy().loc[:, ["name", "optimal", "stderr"]]
        p.loc[:, "name"] = p.index
        stderr = p.loc[:, "stderr"] / p.loc[:, "optimal"]
        p.loc[:, "optimal"] = p.loc[:, "optimal"].apply(
            _table_formatter_params)
        p.loc[:, "stderr"] = stderr.abs().apply(_table_formatter_stderr)

        ax3.axis('off')
        ax3.table(bbox=(0., 0., 1.0, 1.0), cellText=p.values,
                  colWidths=[0.5, 0.25, 0.25], colLabels=p.columns)

        return fig.axes

    @model_tmin_tmax
    def decomposition(self, tmin=None, tmax=None, ytick_base=True, split=True,
                      figsize=(10, 8), axes=None, name=None,
                      return_warmup=False, min_ylim_diff=None, **kwargs):
        """Plot the decomposition of a time-series in the different stresses.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        ytick_base: Boolean or float, optional
            Make the ytick-base constant if True, set this base to float if
            float.
        split: bool, optional
            Split the stresses in multiple stresses when possible. Default is
            True.
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
            Optional arguments, passed on to the plt.subplots method.

        Returns
        -------
        axes: list of matplotlib.axes.Axes
        """
        o = self.ml.observations(tmin=tmin, tmax=tmax)

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax,
                               return_warmup=return_warmup)
        if name is not None:
            sim.name = name

        # determine the influence of the different stresses
        contribs = self.ml.get_contributions(split=split, tmin=tmin, tmax=tmax,
                                             return_warmup=return_warmup)
        names = [s.name for s in contribs]

        if self.ml.transform:
            contrib = self.ml.get_transform_contribution(tmin=tmin, tmax=tmax)
            contribs.append(contrib)
            names.append(self.ml.transform.name)

        # determine ylim for every graph, to scale the height
        ylims = [(min([sim.min(), o[tmin:tmax].min()]),
                  max([sim.max(), o[tmin:tmax].max()]))]
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
                    ylims[i] = (np.mean(ylim) - min_ylim_diff / 2,
                                np.mean(ylim) + min_ylim_diff / 2)
        # determine height ratios
        height_ratios = _get_height_ratios(ylims)

        nrows = len(contribs) + 1
        if axes is None:
            # open a new figure
            gridspec_kw = {'height_ratios': height_ratios}
            fig, axes = plt.subplots(nrows, sharex=True, figsize=figsize,
                                     gridspec_kw=gridspec_kw, **kwargs)
            axes = np.atleast_1d(axes)
            o_label = o.name
            set_axes_properties = True
        else:
            if len(axes) != nrows:
                msg = 'Makes sure the number of axes equals the number of ' \
                      'series'
                raise Exception(msg)
            fig = axes[0].figure
            o_label = ''
            set_axes_properties = False

        # plot simulation and observations in top graph
        o_nu = self.ml.oseries.series.drop(o.index)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(linestyle='', marker='.', color='0.5', label='',
                      markersize=2, ax=axes[0], x_compat=True)
        o.plot(linestyle='', marker='.', color='k', label=o_label,
               markersize=3, ax=axes[0], x_compat=True)
        sim.plot(ax=axes[0], x_compat=True)
        if set_axes_properties:
            axes[0].set_title('observations vs. simulation')
            axes[0].set_ylim(ylims[0])
        axes[0].grid(True)
        axes[0].legend(ncol=3, frameon=False, numpoints=3)

        if ytick_base and set_axes_properties:
            if isinstance(ytick_base, bool):
                # determine the ytick-spacing of the top graph
                yticks = axes[0].yaxis.get_ticklocs()
                if len(yticks) > 1:
                    ytick_base = yticks[1] - yticks[0]
                else:
                    ytick_base = None
            axes[0].yaxis.set_major_locator(
                MultipleLocator(base=ytick_base))

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
        if set_axes_properties:
            axes[0].set_xlim(tmin, tmax)
        fig.tight_layout(pad=0.0)

        return axes

    @model_tmin_tmax
    def diagnostics(self, tmin=None, tmax=None, figsize=(10, 6), bins=50,
                    acf_options=None, fig=None, alpha=0.05, **kwargs):
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
            dictionary with keyword arguments that are passed on to
            pastas.stats.acf.
        fig: Matplotib.Figure instance, optional
            Optionally provide a Matplotib.Figure instance to plot onto.
        alpha: float, optional
            Significance level to calculate the (1-alpha)-confidence intervals.
        **kwargs: dict, optional
            Optional keyword arguments, passed on to plt.figure.

        Returns
        -------
        axes: list of matplotlib.axes.Axes

        Examples
        --------
        >>> axes = ml.plots.diagnostics()

        Note
        ----
        This plot assumed that the noise or residuals follow a Normal
        distribution.

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

        return diagnostics(series=res, figsize=figsize, bins=bins, fig=fig,
                           acf_options=acf_options, alpha=alpha, **kwargs)

    @model_tmin_tmax
    def cum_frequency(self, tmin=None, tmax=None, ax=None, figsize=(5, 2),
                      **kwargs):
        """Plot the cumulative frequency for the observations and simulation.

        Parameters
        ----------
        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        ax: Matplotlib.axes instance, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.
        **kwargs:
            Passed on to plot_cum_frequency

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

    def block_response(self, stressmodels=None, ax=None, figsize=None,
                       **kwargs):
        """Plot the block response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.
        ax: Matplotlib.axes instance, optional
            Axes to add the plot to.
        figsize: tuple, optional
            Tuple with the height and width of the figure in inches.

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
            if hasattr(self.ml.stressmodels[name], 'rfunc'):
                self.ml.get_block_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel %s not in stressmodels list.",
                               name)

        plt.xlim(0)
        plt.xlabel("Time [days]")
        plt.legend(legend)
        return ax

    def step_response(self, stressmodels=None, ax=None, figsize=None,
                      **kwargs):
        """Plot the step response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.

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
            if hasattr(self.ml.stressmodels[name], 'rfunc'):
                self.ml.get_step_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel %s not in stressmodels list.",
                               name)

        plt.xlim(0)
        plt.xlabel("Time [days]")
        plt.legend(legend)
        return ax

    @model_tmin_tmax
    def stresses(self, tmin=None, tmax=None, cols=1, split=True, sharex=True,
                 figsize=(10, 8), **kwargs):
        """This method creates a graph with all the stresses used in the model.

        Parameters
        ----------
        tmin
        tmax
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
        axes: matplotlib.axes
            matplotlib axes instance.
        """
        stresses = _get_stress_series(self.ml, split=split)

        rows = len(stresses)
        rows = -(-rows // cols)  # round up with out additional import

        fig, axes = plt.subplots(rows, cols, sharex=sharex, figsize=figsize,
                                 **kwargs)

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

    @model_tmin_tmax
    def contributions_pie(self, tmin=None, tmax=None, ax=None,
                          figsize=None, split=True, partition='std',
                          wedgeprops=None, startangle=90,
                          autopct='%1.1f%%', **kwargs):
        """Make a pie chart of the contributions. This plot is based on the TNO
        Groundwatertoolbox.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional.
        tmax: str or pandas.Timestamp, optional.
        ax: matplotlib.axes, optional
            Axes to plot the pie chart on. A new figure and axes will be
            created of not providided.
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        partition : str
            statistic to use to determine contribution of stress, either
            'sum' or 'std' (default).
        wedgeprops: dict, optional, default None
            dict containing pie chart wedge properties, default is None,
            which sets edgecolor to white.
        startangle: float
            at which angle to start drawing wedges
        autopct: str
            format string to add percentages to pie chart
        kwargs: dict, optional
            The keyword arguments are passed on to plt.pie.

        Returns
        -------
        ax: matplotlib.axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        contribs = self.ml.get_contributions(split=split, tmin=tmin, tmax=tmax)
        if partition == 'sum':
            # the part of each pie is determined by the sum of the contribution
            frac = [np.abs(contrib).sum() for contrib in contribs]
        elif partition == 'std':
            # the part of each pie is determined by the std of the contribution
            frac = [contrib.std() for contrib in contribs]
        else:
            msg = 'Unknown value for partition: {}'.format(partition)
            raise (Exception(msg))

        # make sure the unexplained part is 100 - evp %
        evp = self.ml.stats.evp(tmin=tmin, tmax=tmax) / 100
        frac = np.array(frac) / sum(frac) * evp
        frac = np.append(frac, 1 - evp)

        if 'labels' not in kwargs:
            labels = [contrib.name for contrib in contribs]
            labels.append("Unexplained")
            kwargs['labels'] = labels

        if wedgeprops is None:
            wedgeprops = {'edgecolor': 'w'}

        ax.pie(frac, wedgeprops=wedgeprops, startangle=startangle,
               autopct=autopct, **kwargs)
        ax.axis('equal')
        return ax

    @model_tmin_tmax
    def stacked_results(self, tmin=None, tmax=None, figsize=(10, 8),
                        stacklegend=False, **kwargs):
        """Create a results plot, similar to `ml.plots.results()`, in which the
        individual contributions of stresses (in stressmodels with multiple
        stresses) are stacked.

        Note: does not plot the individual contributions of StressModel2

        Parameters
        ----------
        tmin : str or pandas.Timestamp, optional
        tmax : str or pandas.Timestamp, optional
        figsize : tuple, optional
        stacklegend : bool, optional
            Add legend to the stacked plot.

        Returns
        -------
        axes: list of axes objects
        """

        # %% Contribution per stress on model results plot
        def custom_sort(t):
            """Sort by mean contribution."""
            return t[1].mean()

        # Create standard results plot
        axes = self.ml.plots.results(tmin=tmin, tmax=tmax, figsize=figsize,
                                     **kwargs)

        nsm = len(self.ml.stressmodels)

        # loop over axes showing stressmodel contributions
        for i, sm in zip(range(3, 3 + 2 * nsm, 2),
                         self.ml.stressmodels.keys()):

            # Get the contributions for StressModels with multiple stresses
            contributions = []
            sml = self.ml.stressmodels[sm]
            if (len(sml.stress) > 0) and (sml._name == "WellModel"):
                nsplit = sml.get_nsplit()
                if nsplit > 1:
                    for istress in range(len(sml.stress)):
                        h = self.ml.get_contribution(sm, istress=istress,
                                                     tmin=tmin, tmax=tmax)
                        name = sml.stress[istress].name
                        if name is None:
                            name = sm
                        contributions.append((name, h))
                else:
                    h = self.ml.get_contribution(sm, tmin=tmin, tmax=tmax)
                    name = sm
                    contributions.append((name, h))
                contributions.sort(key=custom_sort)

                # add stacked plot to correct axes
                ax = axes[i - 1]
                del ax.lines[0]  # delete existing line

                contrib = [c[1] for c in contributions]  # get timeseries
                vstack = concat(contrib, axis=1, sort=False)
                names = [c[0] for c in contributions]  # get names
                ax.stackplot(vstack.index, vstack.values.T, labels=names)
                if stacklegend:
                    ax.legend(loc="best", ncol=5, fontsize=8)

                # y-scale does not show 0
                ylower, yupper = ax.get_ylim()
                if (ylower < 0) and (yupper < 0):
                    ax.set_ylim(top=0)
                elif (ylower > 0) and (yupper > 0):
                    ax.set_ylim(bottom=0)

        return axes

    @model_tmin_tmax
    def series(self, tmin=None, tmax=None, split=True, **kwargs):
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
        labels: List of str
            List with the labels for each subplot.
        figsize: tuple
            Set the size of the figure.

        Returns
        -------
        matplotlib.Axes
        """
        obs = self.ml.observations(tmin=tmin, tmax=tmax)
        stresses = _get_stress_series(self.ml, split=split)
        axes = series(obs, stresses=stresses, **kwargs)
        return axes

    @model_tmin_tmax
    def summary_pdf(self, tmin=None, tmax=None, fname=None, dpi=150,
                    results_kwargs={}, diagnostics_kwargs={}):
        """Create a PDF file (A4) with the results and diagnostics plot.

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
        fig: matplotlib.Figure instance

        """
        if fname is None:
            fname = "{}.pdf".format(self.ml.name)
        pdf = PdfPages(fname)

        fig = plt.figure(figsize=(8.27, 11.69), dpi=50)

        fig1, fig2 = fig.subfigures(2, 1, height_ratios=[1.25, 1.])

        self.results(fig=fig1, tmin=tmin, tmax=tmax, **results_kwargs)
        self.diagnostics(fig=fig2, tmin=tmin, tmax=tmax, **diagnostics_kwargs)
        fig2.subplots_adjust(wspace=0.2)

        fig1.suptitle("Model Results", fontweight="bold")
        fig2.suptitle("Model Diagnostics", fontweight="bold")

        plt.subplots_adjust(left=0.1, top=0.9, right=0.95, bottom=0.1)
        pdf.savefig(fig, papertype="a4", orientation="portrait", dpi=dpi)
        pdf.close()
        return fig


def _get_height_ratios(ylims):
    height_ratios = []
    for ylim in ylims:
        hr = ylim[1] - ylim[0]
        if np.isnan(hr):
            hr = 0.0
        height_ratios.append(hr)
    return height_ratios


def _get_stress_series(ml, split=True):
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
