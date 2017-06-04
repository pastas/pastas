"""
This file contains the plotting functionalities that are available for Pastas.

Examples
--------
    ml.plot.decomposition()

"""

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from scipy.stats import probplot
import numpy as np
from .utils import get_dt

class Plotting():
    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that are available."
        return msg

    def plot(self, tmin=None, tmax=None, oseries=True, simulate=True, figsize=None, show=True):
        """

        Parameters
        ----------
        oseries: Boolean
            True to plot the observed time series.

        Returns
        -------
        Plot of the simulated and optionally the observed time series

        """
        plt.figure()
        plt.title(self.ml.name)
        if oseries:
            self.ml.oseries.plot(linestyle='', marker='.', color='k',
                                 markersize=3, figsize=figsize)
        if simulate:
            if tmin is None:
                tmin = self.ml.oseries.index.min()
            if tmax is None:
                tmax = self.ml.oseries.index.max()
            h = self.ml.simulate(tmin=tmin, tmax=tmax)
            h.plot(figsize=figsize)

        if show: plt.show()

    def results(self, tmin=None, tmax=None, savefig=False, figsize=None):
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin/tmax: str
            start and end time for plotting
        savefig: boolean, optional
            True to save the figure, False is default. Figure is saved in the
            current working directory when running your python scripts.

        Returns
        -------

        """
        fig = plt.figure(facecolor='white', figsize=figsize)
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
        for name, ts in self.ml.tseriesdict.items():
            dt = get_dt(self.ml.freq)
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
        plt.text(0.05, 0.8, 'AIC: %.2f' % self.ml.stats.aic())
        plt.text(0.05, 0.6, 'BIC: %.2f' % self.ml.stats.aic())
        plt.title('Statistics', loc='left')
        plt.show()
        if savefig:
            plt.savefig('pastas.eps', bbox_inches='tight')

    def decomposition(self, tmin=None, tmax=None):
        """Plot the decomposition of a time-series in the different stresses.

        """

        # Default option when not tmin and tmax is provided
        if tmin is None:
            tmin = self.ml.tmin
        if tmax is None:
            tmax = self.ml.tmax
        assert (tmin is not None) and (
            tmax is not None), 'model needs to be solved first'

        # determine the simulation
        hsim = self.ml.simulate(tmin=tmin, tmax=tmax)
        tindex = hsim.index
        h = [hsim]

        # determine the influence of the different stresses
        for ts in self.ml.tseriesdict.values():
            h.append(self.ml.get_contribution(ts.name, tindex=tindex))

        # open the figure
        if False:
            f, axarr = plt.subplots(1 + len(self.ml.tseriesdict), sharex=True)
        else:
            # let the height of the axes be determined by the values
            # height_ratios = [1]*(len(self.tseriesdict)+1)
            height_ratios = [max([hsim.max(), self.ml.oseries.max()]) - min(
                [hsim.min(), self.ml.oseries.min()])]
            for ht in h[1:]:
                height_ratios.append(ht.max() - ht.min())
            f, axarr = plt.subplots(1 + len(self.ml.tseriesdict), sharex=True,
                                    gridspec_kw={
                                        'height_ratios': height_ratios})
            axarr = np.atleast_1d(axarr)

        # plot simulation and observations in top graph
        plt.axes(axarr[0])
        self.ml.oseries.plot(linestyle='', marker='.', color='k', markersize=3,
                             ax=axarr[0], label='observations')
        hsim.plot(ax=axarr[0], label='simulation')
        axarr[0].autoscale(enable=True, axis='y', tight=True)
        axarr[0].grid(which='both')
        axarr[0].minorticks_off()

        # add a legend
        axarr[0].legend(loc=(0, 1), ncol=3, frameon=False)

        # determine the ytick-spacing of the top graph
        yticks, ylabels = plt.yticks()
        if len(yticks) > 2:
            base = yticks[1] - yticks[0]
        else:
            base = None

        # plot the influence of the stresses
        iax = 1
        for ts in self.ml.tseriesdict.values():
            plt.axes(axarr[iax])
            plt.plot(h[iax].index, h[iax].values)
            if base is not None:
                # set the ytick-spacing equal to the top graph
                axarr[iax].yaxis.set_major_locator(
                    plticker.MultipleLocator(base=base))

            axarr[iax].set_title(ts.name)
            axarr[iax].autoscale(enable=True, axis='y', tight=True)
            axarr[iax].grid(which='both')
            axarr[iax].minorticks_off()
            iax += 1

        # show the figure
        plt.tight_layout()
        plt.show()

    def diagnostics(self, tmin=None, tmax=None):
        innovations = self.ml.innovations(tmin, tmax)

        plt.figure()
        gs = plt.GridSpec(2, 3, wspace=0.2)

        plt.subplot(gs[0, :2])
        plt.title('Autocorrelation')
        # plt.axhline(0.2, '--')
        plt.stem(self.ml.stats.acf())

        plt.subplot(gs[1, :2])
        plt.title('Partial Autocorrelation')
        # plt.axhline(0.2, '--')
        plt.stem(self.ml.stats.pacf())

        plt.subplot(gs[0, 2])
        innovations.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(innovations, plot=plt)
        plt.show()

    def block_response(self, series=None):
        """Plot the block response for a specific series.

        Returns
        -------
        fig: matplotlib.Figure
            return a Matplotlib figure instance.

        """
        if not series:
            series = self.ml.tseriesdict.keys()
        else:
            series = [series]

        legend = []
        fig = plt.figure()

        for name in series:
            if name not in self.ml.tseriesdict.keys():
                return None
            elif hasattr(self.ml.tseriesdict[name], 'rfunc'):
                plt.plot(self.ml.get_block_response(name))
                legend.append(name)
            else:
                pass

        plt.xlim(0)

        # Change xtickers to the correct time
        locs, labels = plt.xticks()
        labels = locs * get_dt(self.ml.freq)
        plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Block Response(s)")
        return fig

    def step_response(self, series=None):
        """Plot the step response for a specific series.

        Returns
        -------
        fig: matplotlib.Figure
            return a Matplotlib figure instance.

        """
        if not series:
            series = self.ml.tseriesdict.keys()
        else:
            series = [series]

        legend = []
        fig = plt.figure()

        for name in series:
            if name not in self.ml.tseriesdict.keys():
                return None
            elif hasattr(self.ml.tseriesdict[name], 'rfunc'):
                plt.plot(self.ml.get_step_response(name))
                legend.append(name)
            else:
                pass

        plt.xlim(0)

        # Change xtickers to the correct time
        locs, labels = plt.xticks()
        labels = locs * get_dt(self.ml.freq)
        plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Step Response(s)")
        return fig
