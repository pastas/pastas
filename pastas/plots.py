"""
This file contains the plotting functionalities that are available for Pastas.

Examples
--------
    ml.plot.decomposition()

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pastas as ps
from scipy.stats import probplot

from .decorators import model_tmin_tmax
from .utils import get_dt


class Plotting():
    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that are " \
              "available."
        return msg

    @model_tmin_tmax
    def plot(self, tmin=None, tmax=None, oseries=True, simulation=True,
             show=True, **kwargs):
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        oseries: Boolean
            True to plot the observed time series.
        simulation: Boolean
            True to plot the simulated time series.

        Returns
        -------
        fig: matplotlib.figure
            MPL figure with the simulated and optionally the observed time
            series.

        """
        fig = self._get_figure(**kwargs)
        fig.suptitle("Results of " + self.ml.name)

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o.plot(linestyle='', marker='.', color='k', fig=fig)

        if simulation:
            h = self.ml.simulate(tmin=tmin, tmax=tmax)
            h.plot(fig=fig)

        plt.ylabel("Groundwater levels [meter]")
        plt.legend()

        if show:
            plt.show()

        return fig.axes

    @model_tmin_tmax
    def results(self, tmin=None, tmax=None, show=True):
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin/tmax: str
            start and end time for plotting


        Returns
        -------

        """
        fig = self._get_figure()

        #Number of row to make the figure with
        rows = 3 + len(self.ml.stressmodels)

        # Main frame
        ax1 = plt.subplot2grid((rows, 3), (0, 0), colspan=2, rowspan=2)
        o = self.ml.observations(tmin=tmin, tmax=tmax)
        o.plot(ax=ax1, linestyle='',  marker='.', color='k', x_compat=True)
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        sim.plot(ax=ax1, x_compat=True)
        plt.legend(loc=(0, 1), ncol=3, frameon=False)

        # Residuals and innovations
        ax2 = plt.subplot2grid((rows, 3), (2, 0), colspan=2, sharex=ax1)
        res = self.ml.residuals(tmin=tmin, tmax=tmax)
        res.plot(ax=ax2, sharex=ax1, color='k', x_compat=True)
        if self.ml.settings["noise"]:
            v = self.ml.innovations(tmin=tmin, tmax=tmax)
            v.plot(ax=ax2, sharex=ax1, x_compat=True)
        plt.legend(loc=(0, 1), ncol=3, frameon=False)

        # Stats frame
        ax3 = plt.subplot2grid((rows, 3), (0, 2), rowspan=3)
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        plt.text(0.05, 0.8, 'Rsq: %.2f' % self.ml.stats.rsq())
        plt.text(0.05, 0.6, 'EVP: %.2f' % self.ml.stats.evp())
        plt.title('Model Information', loc='left')

        # Add a row for each stressmodel
        for i, ts in enumerate(self.ml.stressmodels.keys(), start=3):
            ax = plt.subplot2grid((rows, 3), (i, 0), colspan=2, sharex=ax1)
            contrib = self.ml.get_contribution(ts, tmin=tmin, tmax=tmax)
            contrib.plot(ax=ax, sharex=ax1, x_compat=True)
            title = [stress.name for stress in self.ml.stressmodels[ts].stress]
            plt.title("Stresses:%s" % title, loc="right")
            ax.legend(loc=(0, 1), ncol=3, frameon=False)
            axb = plt.subplot2grid((rows, 3), (i, 2))
            self.ml.get_step_response(ts).plot(ax=axb)

        if show:
            plt.show()

        return fig.axes

    @model_tmin_tmax
    def decomposition(self, tmin=None, tmax=None, show=True, ytick_base=True):
        """Plot the decomposition of a time-series in the different stresses.

        """
        # determine the simulation
        hsim = self.ml.simulate(tmin=tmin, tmax=tmax)
        tindex = hsim.index
        h = [hsim]

        # determine the influence of the different stresses
        for name in self.ml.stressmodels.keys():
            h.append(self.ml.get_contribution(name, tindex=tindex))

        # open the figure
        height_ratios = [max([hsim.max(), self.ml.oseries.max()]) - min(
            [hsim.min(), self.ml.oseries.min()])]
        for ht in h[1:]:
            hr = ht.max() - ht.min()
            if np.isnan(hr):
                hr = 0.0
            height_ratios.append(hr)

        fig, ax = plt.subplots(1 + len(self.ml.stressmodels), sharex=True,
                               gridspec_kw={'height_ratios': height_ratios})
        ax = np.atleast_1d(ax)

        # plot simulation and observations in top graph
        self.ml.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                             ax=ax[0], x_compat=True)
        hsim.plot(ax=ax[0], x_compat=True)
        ax[0].autoscale(enable=True, axis='y', tight=True)
        ax[0].grid(which='both')
        ax[0].legend(loc=(0, 1), ncol=3, frameon=False)

        if ytick_base:
            if isinstance(ytick_base, bool):
                # determine the ytick-spacing of the top graph
                yticks = ax[0].yaxis.get_ticklocs()
                if len(yticks) > 1:
                    ytick_base = yticks[1] - yticks[0]
                else:
                    ytick_base = None
            ax[0].yaxis.set_major_locator(
                plticker.MultipleLocator(base=ytick_base))

        # plot the influence of the stresses
        for i, name in enumerate(self.ml.stressmodels.keys(), start=1):
            h[i].plot(ax=ax[i], x_compat=True)

            if ytick_base:
                # set the ytick-spacing equal to the top graph
                ax[i].yaxis.set_major_locator(
                    plticker.MultipleLocator(base=ytick_base))

            ax[i].set_title(name)
            ax[i].autoscale(enable=True, axis='y', tight=True)
            ax[i].grid(which='both')
            ax[i].minorticks_off()

        if show:
            plt.show()

        return fig.axes

    @model_tmin_tmax
    def diagnostics(self, tmin=None, tmax=None, show=True):
        innovations = self.ml.innovations(tmin, tmax)

        fig = self._get_figure()
        gs = plt.GridSpec(2, 3, wspace=0.2)

        plt.subplot(gs[0, :2])
        plt.title('Autocorrelation')
        # plt.axhline(0.2, '--')
        r = ps.stats.acf(innovations)
        plt.stem(r)

        plt.subplot(gs[1, :2])
        plt.title('Partial Autocorrelation')
        # plt.axhline(0.2, '--')
        # plt.stem(self.ml.stats.pacf())

        plt.subplot(gs[0, 2])
        innovations.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(innovations, plot=plt)

        if show:
            plt.show()

        return fig.axes

    def block_response(self, series=None, show=True):
        """Plot the block response for a specific series.

        Returns
        -------
        fig: matplotlib.Figure
            return a Matplotlib figure instance.

        """
        if not series:
            series = self.ml.stressmodels.keys()
        else:
            series = [series]

        legend = []
        fig = self._get_figure()

        for name in series:
            if name not in self.ml.stressmodels.keys():
                return None
            elif hasattr(self.ml.stressmodels[name], 'rfunc'):
                plt.plot(self.ml.get_block_response(name))
                legend.append(name)
            else:
                pass

        plt.xlim(0)

        # Change xtickers to the correct time
        locs, labels = plt.xticks()
        labels = locs * get_dt(self.ml.settings["freq"])
        plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Block Response(s)")

        if show:
            plt.show()

        return fig.axes

    def step_response(self, series=None, show=True):

        """Plot the step response for a specific series.

        Returns
        -------
        fig: matplotlib.Figure
            return a Matplotlib figure instance.

        """
        if not series:
            series = self.ml.stressmodels.keys()
        else:
            series = [series]

        legend = []
        fig = self._get_figure()

        for name in series:
            if name not in self.ml.stressmodels.keys():
                return None
            elif hasattr(self.ml.stressmodels[name], 'rfunc'):
                plt.plot(self.ml.get_step_response(name))
                legend.append(name)
            else:
                pass

        plt.xlim(0)

        # Change xtickers to the correct time
        locs, labels = plt.xticks()
        labels = locs * get_dt(self.ml.settings["freq"])
        plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Step Response(s)")

        if show:
            plt.show()

        return fig.axes

    @model_tmin_tmax
    def stresses(self, tmin=None, tmax=None, cols=1, **kwargs):
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
            matplotlib aes instance.

        """
        stresses = []

        for name in self.ml.stressmodels.keys():
            stress = self.ml.get_stress(name)
            if isinstance(stress, list):
                stresses.extend(stress)
            else:
                stresses.append(stress)

        rows = len(stresses)
        rows = -(-rows // cols)  # round up with out additional import

        fig, axes = plt.subplots(rows, cols, **kwargs)

        if hasattr(axes, "flatten"):
            axes = axes.flatten()
        else:
            axes = [axes]

        for ax, stress in zip(axes, stresses):
            stress.plot(ax=ax, x_compat=True)
            ax.legend([stress.name], loc=2)

        return axes

    def _get_figure(self, **kwargs):
        fig = plt.figure(**kwargs)
        return fig
