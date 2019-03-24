"""
This file contains the plotting functionalities that are available for Pastas.

Examples
--------
    ml.plot.decomposition()

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import probplot

from .decorators import model_tmin_tmax
from .stats import acf

import logging

logger = logging.getLogger(__name__)


class Plotting:
    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that " \
              "are available."
        return msg

    @model_tmin_tmax
    def plot(self, tmin=None, tmax=None, oseries=True, simulation=True,
             **kwargs):
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        oseries: bool, optional
            True to plot the observed time series.
        simulation: bool, optional
            True to plot the simulated time series.

        Returns
        -------
        ax: matplotlib.axes
            matplotlib axes with the simulated and optionally the observed
            timeseries.

        """
        ax = plt.subplot(**kwargs)
        ax.set_title("Results of {}".format(self.ml.name))

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o_nu = self.ml.oseries.series.drop(o.index)
            if not o_nu.empty:
                # plot parts of the oseries that are not used in grey
                o_nu.plot(linestyle='', marker='.', color='0.5', label='',
                          ax=ax)
            o.plot(linestyle='', marker='.', color='k', ax=ax)

        if simulation:
            sim = self.ml.simulate(tmin=tmin, tmax=tmax)
            sim.plot(ax=ax)
        plt.xlim(tmin, tmax)
        plt.ylabel("Groundwater levels [meter]")
        plt.legend()
        plt.tight_layout()
        return ax

    @model_tmin_tmax
    def results(self, tmin=None, tmax=None, figsize=(10, 8), **kwargs):
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.

        Returns
        -------
        matplotlib.axes

        """
        # Number of rows to make the figure with
        rows = 3 + len(self.ml.stressmodels)
        fig = plt.figure(figsize=figsize, **kwargs)
        # Main frame
        ax1 = plt.subplot2grid((rows, 3), (0, 0), colspan=2, rowspan=2)
        o = self.ml.observations(tmin=tmin, tmax=tmax)
        o_nu = self.ml.oseries.series.drop(o.index)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(ax=ax1, linestyle='', marker='.', color='0.5', label='',
                      x_compat=True)
        o.plot(ax=ax1, linestyle='', marker='.', color='k', x_compat=True)
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        sim.plot(ax=ax1, x_compat=True)
        ax1.legend(loc=(0, 1), ncol=3, frameon=False)
        ax1.set_ylim(min(o.min(), sim.loc[tmin:tmax].min()),
                     max(o.max(), sim.loc[tmin:tmax].max()))
        ax1.minorticks_off()

        # Residuals and noise
        ax2 = plt.subplot2grid((rows, 3), (2, 0), colspan=2, sharex=ax1)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        res.plot(ax=ax2, sharex=ax1, color='k', x_compat=True)
        if self.ml.settings["noise"] and self.ml.noisemodel:
            noise = self.ml.noise(tmin=tmin, tmax=tmax)
            noise.plot(ax=ax2, sharex=ax1, x_compat=True)
        ax2.axhline(0.0, color='k', linestyle='--', zorder=0)
        ax2.legend(loc=(0, 1), ncol=3, frameon=False)
        ax2.minorticks_off()

        # Stats frame
        ax3 = plt.subplot2grid((rows, 3), (0, 2), rowspan=3)
        ax3.set_title('Model Information', loc='left')

        # Add a row for each stressmodel
        for i, sm in enumerate(self.ml.stressmodels.keys(), start=3):
            ax = plt.subplot2grid((rows, 3), (i, 0), colspan=2, sharex=ax1)
            contrib = self.ml.get_contribution(sm, tmin=tmin, tmax=tmax)
            contrib.plot(ax=ax, sharex=ax1, x_compat=True)
            title = [stress.name for stress in self.ml.stressmodels[sm].stress]
            plt.title("Stresses:%s" % title, loc="right")
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
            if i == 3:
                sharex = None
            else:
                sharex = axb
            axb = plt.subplot2grid((rows, 3), (i, 2), sharex=sharex)
            self.ml.get_step_response(sm).plot(ax=axb)
            ax.minorticks_off()

        ax1.set_xlim(tmin, tmax)

        plt.tight_layout(pad=0.0)

        # Draw parameters table
        parameters = self.ml.parameters.copy()
        parameters['name'] = parameters.index
        cols = ["name", "optimal", "stderr"]
        parameters = parameters.loc[:, cols]
        for name, vals in parameters.loc[:, cols].iterrows():
            parameters.loc[name, "optimal"] = '{:.2f}'.format(vals.optimal)
            stderr_perc = np.abs(np.divide(vals.stderr, vals.optimal) * 100)
            parameters.loc[name, "stderr"] = '{:.1f}{}'.format(stderr_perc,
                                                               "\u0025")
        ax3.axis('off')
        # loc='upper center'
        ax3.table(bbox=(0., 0., 1.0, 1.0), cellText=parameters.values,
                  colWidths=[0.5, 0.25, 0.25], colLabels=cols)

        return fig.axes

    @model_tmin_tmax
    def decomposition(self, tmin=None, tmax=None, ytick_base=True, split=True,
                      figsize=(10, 8), axes=None, name=None,
                      return_warmup=False, **kwargs):
        """Plot the decomposition of a time-series in the different stresses.

        Parameters
        ----------
        tmin: str or pandas.Timestamp, optional
        tmax: str or pandas.Timestamp, optional
        ytick_base: Boolean or float, optional
            Make the ytick-base constant if True, set this base to float if
            float.
        split: bool, optional
            Split the stresses in multiple stresses when possible.
        axes: matplotlib.Axes instance, optional
            Matplotlib Axes instance to plot the figure on to.
        figsize: tuple, optional
            tuple of size 2 to determine the figure size in inches.
        name: str, optional
            Name to give the simulated time series in the legend.
        **kwargs: dict, optional
            Optional arguments, passed on to the plt.subplots method.

        Returns
        -------
        axes: list of matplotlib.axes

        """
        o = self.ml.observations(tmin=tmin, tmax=tmax)

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax,
                               return_warmup=return_warmup)
        if name is not None:
            sim.name = name
        series = [sim]
        names = ['']

        # determine the influence of the different stresses
        for name in self.ml.stressmodels.keys():
            nstress = len(self.ml.stressmodels[name].stress)
            if split and nstress > 1:
                for istress in range(nstress):
                    contrib = self.ml.get_contribution(
                        name, tmin=tmin, tmax=tmax, istress=istress,
                        return_warmup=return_warmup
                    )
                    series.append(contrib)
                    names.append(contrib.name)
            else:
                contrib = self.ml.get_contribution(
                    name, tmin=tmin, tmax=tmax, return_warmup=return_warmup
                )

                series.append(contrib)
                names.append(contrib.name)

        if self.ml.transform:
            series.append(
                self.ml.get_transform_contribution(tmin=tmin, tmax=tmax))
            names.append(self.ml.transform.name)

        # determine ylim for every graph, to scale the height
        ylims = [
            (min([sim[tmin:tmax].min(), o[tmin:tmax].min()]),
             max([sim[tmin:tmax].max(), o[tmin:tmax].max()]))]
        for contrib in series[1:]:
            hs = contrib[tmin:tmax]
            if hs.empty:
                if contrib.empty:
                    ylims.append((0.0, 0.0))
                else:
                    ylims.append((contrib.min(), hs.max()))
            else:
                ylims.append((hs.min(), hs.max()))
        height_ratios = [
            0.0 if np.isnan(ylim[1] - ylim[0]) else ylim[1] - ylim[0] for ylim
            in ylims]

        if axes is None:
            # open a new figure
            fig, axes = plt.subplots(len(series), sharex=True, figsize=figsize,
                                     gridspec_kw={
                                         'height_ratios': height_ratios},
                                     **kwargs)
            axes = np.atleast_1d(axes)
            o_label = o.name
            set_axes_properties = True
        else:
            assert len(axes) == len(series), 'Makes sure the number of axes ' \
                                             'equals the number of series'
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
        axes[0].grid(which='both')
        axes[0].legend(ncol=3, frameon=False)

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
        for i, contrib in enumerate(series[1:], start=1):
            contrib.plot(ax=axes[i], x_compat=True)
            if set_axes_properties:
                if ytick_base:
                    # set the ytick-spacing equal to the top graph
                    axes[i].yaxis.set_major_locator(
                        MultipleLocator(base=ytick_base))

                axes[i].set_title(names[i])
                axes[i].set_ylim(ylims[i])
            axes[i].grid(which='both')
            axes[i].minorticks_off()
        if set_axes_properties:
            axes[0].set_xlim(tmin, tmax)
        fig.tight_layout(pad=0.0)

        return axes

    @model_tmin_tmax
    def diagnostics(self, tmin=None, tmax=None):
        """Plot a window that helps in diagnosing basic model assumptions.

        Parameters
        ----------
        tmin
        tmax

        Returns
        -------
        matplotlib.axes

        """
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax)
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)

        shape = (2, 3)
        ax = plt.subplot2grid(shape, (0, 0), colspan=2, rowspan=1)
        ax.set_title(res.name)
        res.plot(ax=ax)

        ax1 = plt.subplot2grid(shape, (1, 0), colspan=2, rowspan=1)
        ax1.set_ylabel('Autocorrelation')
        conf = 1.96 / np.sqrt(res.index.size)
        r = acf(res)

        ax1.axhline(conf, linestyle='--', color="dimgray")
        ax1.axhline(-conf, linestyle='--', color="dimgray")
        ax1.stem(r.index, r.values, basefmt="gray")
        ax1.set_xlim(r.index.min(), r.index.max())
        ax1.set_xlabel("Lag (Days)")

        ax2 = plt.subplot2grid(shape, (0, 2), colspan=1, rowspan=1)
        res.hist(bins=20, ax=ax2)

        ax3 = plt.subplot2grid(shape, (1, 2), colspan=1, rowspan=1)
        probplot(res, plot=ax3, dist="norm", rvalue=True)

        c = ax.get_lines()[0]._color
        ax3.get_lines()[0].set_color(c)

        plt.tight_layout(pad=0.0)
        return plt.gca()

    def block_response(self, stressmodels=None, **kwargs):
        """Plot the block response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.

        Returns
        -------
        matplotlib.axes
            matplotlib axes instance.

        """
        if not stressmodels:
            stressmodels = self.ml.stressmodels.keys()

        legend = []

        ax = plt.subplot(**kwargs)

        for name in stressmodels:
            if hasattr(self.ml.stressmodels[name], 'rfunc'):
                self.ml.get_block_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel {} not in stressmodels "
                               "list.".format(name))

        plt.xlim(0)
        plt.xlabel("Time [days]")
        plt.legend(legend)
        return ax

    def step_response(self, stressmodels=None, **kwargs):
        """Plot the block response for a specific stressmodels.

        Parameters
        ----------
        stressmodels: list, optional
            List with the stressmodels to plot the block response for.

        Returns
        -------
        matplotlib.axes
            matplotlib axes instance.

        """
        if not stressmodels:
            stressmodels = self.ml.stressmodels.keys()

        legend = []

        ax = plt.subplot(**kwargs)

        for name in stressmodels:
            if hasattr(self.ml.stressmodels[name], 'rfunc'):
                self.ml.get_step_response(name).plot(ax=ax)
                legend.append(name)
            else:
                logger.warning("Stressmodel {} not in stressmodels "
                               "list.".format(name))

        plt.xlim(0)
        plt.xlabel("Time [days]")
        plt.legend(legend)
        return ax

    @model_tmin_tmax
    def stresses(self, tmin=None, tmax=None, cols=1, split=True, sharex=True,
                 figsize=(10, 8), **kwargs):
        """This method creates a graph with all the stresses used in the
         model.

        Parameters
        ----------
        tmin
        tmax
        cols: int
            number of columns used for plotting.

        Returns
        -------
        axes: matplotlib.axes
            matplotlib axes instance.

        """
        stresses = []

        for name in self.ml.stressmodels.keys():
            nstress = len(self.ml.stressmodels[name].stress)
            if split and nstress > 1:
                for istress in range(nstress):
                    stress = self.ml.get_stress(name, istress=istress)
                    stresses.append(stress)
            else:
                stress = self.ml.get_stress(name)
                if isinstance(stress, list):
                    stresses.extend(stress)
                else:
                    stresses.append(stress)

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
    def contributions_pie(self, tmin=None, tmax=None, ax=None, **kwargs):
        """Make a pie chart of the contributions. This plot is based on the
        TNO Groundwatertoolbox.

        Parameters
        ----------
        tmin
        tmax
        ax: matplotlib.axes, optional
            Axes to plot the pie chart on. A new figure and axes will be
            created of not providided.
        kwargs: dict, optional
            The keyword arguments are passed on to plt.pie.

        Returns
        -------
        ax: matplotlib.axes

        """
        if ax is None:
            _, ax = plt.subplots()

        frac = []
        for name in self.ml.stressmodels.keys():
            frac.append(np.abs(self.ml.get_contribution(name, tmin=tmin,
                                                        tmax=tmax)).sum())

        evp = self.ml.stats.evp(tmin=tmin) / 100
        frac = np.array(frac) / sum(frac) * evp
        frac = frac.tolist()
        frac.append(1 - evp)
        frac = np.array(frac)
        labels = list(self.ml.stressmodels.keys())
        labels.append("Unexplained")
        ax.pie(frac, labels=labels, autopct='%1.1f%%', startangle=90,
               wedgeprops=dict(width=1, edgecolor='w'), **kwargs)
        ax.axis('equal')
        return ax
