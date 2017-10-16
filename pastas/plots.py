"""
This file contains the plotting functionalities that are available for Pastas.

Examples
--------
    ml.plot.decomposition()

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from scipy.stats import probplot

import pastas as ps
from .utils import get_dt


class Plotting():
    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that are available."
        return msg

    def plot(self, tmin=None, tmax=None, show=True, oseries=True,
             simulate=True, **kwargs):
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        oseries: Boolean
            True to plot the observed time series.
        simulate: Boolean
            True to plot the simulated time series.

        Returns
        -------
        fig: matplotlib.figure
            MPL figure with the simulated and optionally the observed time
            series.

        """
        fig = self._get_figure(**kwargs)
        fig.suptitle("Results of " + self.ml.name)

        # Get right tmin and tmax
        if not tmin and not tmax:
            tmin, tmax = self.ml.get_tmin_tmax(use_oseries=oseries)
        elif not tmin:
            tmin = self.ml.get_tmin_tmax(use_oseries=oseries)[0]
        elif not tmax:
            tmin = self.ml.get_tmin_tmax(use_oseries=oseries)[1]

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o.plot(linestyle='', marker='.', color='k', fig=fig)

        if simulate:
            h = self.ml.simulate(tmin=tmin, tmax=tmax)
            h.plot(fig=fig)

        plt.xlabel("Time [days]")
        plt.ylabel("Groundwater levels [meter]")
        plt.legend()

        if show:
            plt.show()

        return fig.axes

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

        gs = plt.GridSpec(3, 4, wspace=0.4, hspace=0.4)

        # Plot the Groundwater levels
        h = self.ml.simulate(tmin=tmin, tmax=tmax)
        ax1 = plt.subplot(gs[:2, :-1])
        self.ml.oseries.plot(linestyle='', marker='.', color='k',
                             markersize=3, label='observed head', ax=ax1)
        h.plot(label='modeled head', ax=ax1)
        ax1.grid(which='both')
        ax1.minorticks_off()
        plt.legend(loc=(0, 1), ncol=3, frameon=False)
        plt.ylabel('Head [m]')

        # Plot the residuals and innovations
        residuals = self.ml.residuals(tmin=tmin, tmax=tmax)
        ax2 = plt.subplot(gs[2, :-1], sharex=ax1)
        residuals.plot(color='k', label='residuals')
        if self.ml.noisemodel is not None:
            innovations = self.ml.innovations(tmin=tmin, tmax=tmax)
            innovations.plot(label='innovations')
        ax2.grid(which='both')
        ax2.minorticks_off()
        plt.legend(loc=(0, 1), ncol=3, frameon=False)
        plt.ylabel('Error [m]')
        plt.xlabel('Time [Years]')

        # Plot the block response function
        ax3 = plt.subplot(gs[0, -1])
        tmax = 0
        for name, ts in self.ml.stressmodels.items():
            dt = get_dt(self.ml.settings["freq"])
            if "rfunc" in dir(ts):
                br = self.ml.get_block_response(name)
                t = np.arange(0, len(br) * dt, dt)
                tmax = max(t[-1], tmax)
                ax3.plot(t, br)
        ax3.set_xlim(0, tmax)
        ax3.grid(which='both')
        ax3.set_title('Block Response', loc='left')

        # Table of the numerical diagnostic statistics.
        ax5 = plt.subplot(gs[2, -1])
        ax5.xaxis.set_visible(False)
        ax5.yaxis.set_visible(False)
        plt.text(0.05, 0.8, 'Rsq: %.2f' % self.ml.stats.rsq())
        plt.text(0.05, 0.6, 'EVP: %.2f' % self.ml.stats.evp())
        plt.title('Statistics', loc='left')

        if show:
            plt.show()

        return fig.axes

    def decomposition(self, tmin=None, tmax=None, show=True):
        """Plot the decomposition of a time-series in the different stresses.

        """

        # Default option when not tmin and tmax is provided
        if tmin is None:
            tmin = self.ml.settings["tmin"]
        if tmax is None:
            tmax = self.ml.settings["tmax"]
        assert (tmin is not None) and (
            tmax is not None), 'model needs to be solved first'

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
        ax = np.atleast_1d(ax)  # ax.Flatten is maybe better?

        # plot simulation and observations in top graph
        self.ml.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                             ax=ax[0], label='observations', x_compat=True)
        hsim.plot(ax=ax[0], label='simulation', x_compat=True)
        ax[0].autoscale(enable=True, axis='y', tight=True)
        ax[0].grid(which='both')
        ax[0].minorticks_off()
        ax[0].legend(loc=(0, 1), ncol=3, frameon=False)

        # determine the ytick-spacing of the top graph
        yticks, ylabels = plt.yticks()
        if len(yticks) > 2:
            base = yticks[1] - yticks[0]
        else:
            base = None

        # plot the influence of the stresses
        for i, name in enumerate(self.ml.stressmodels.keys(), start=1):
            h[i].plot(ax=ax[i], x_compat=True)

            if base is not None:
                # set the ytick-spacing equal to the top graph
                ax[i].yaxis.set_major_locator(
                    plticker.MultipleLocator(base=base))

            ax[i].set_title(name)
            ax[i].autoscale(enable=True, axis='y', tight=True)
            ax[i].grid(which='both')
            ax[i].minorticks_off()

        if show:
            plt.show()

        return fig.axes

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

    def _get_figure(self, **kwargs):
        fig = plt.figure(**kwargs)
        return fig
