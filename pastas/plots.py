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


class Plotting:
    def __init__(self, ml):
        self.ml = ml  # Store a reference to the model class

    def __repr__(self):
        msg = "This module contains all the built-in plotting options that are " \
              "available."
        return msg

    @model_tmin_tmax
    def plot(self, tmin=None, tmax=None, oseries=True, simulation=True,
             **kwargs):
        """Make a plot of the observed and simulated series.

        Parameters
        ----------
        oseries: Boolean
            True to plot the observed time series.
        simulation: Boolean
            True to plot the simulated time series.

        Returns
        -------
        ax: matplotlib.axes
            matplotlib axes with the simulated and optionally the observed time series.

        """
        fig = self._get_figure(**kwargs)
        fig.suptitle("Results of " + self.ml.name)

        if oseries:
            o = self.ml.observations(tmin=tmin, tmax=tmax)
            o_nu = self.ml.oseries.series.drop(o.index)
            if not o_nu.empty:
                # plot parts of the oseries that are not used in grey
                o_nu.plot(linestyle='', marker='.', color='0.5', fig=fig,
                          label='')
            o.plot(linestyle='', marker='.', color='k', fig=fig)

        if simulation:
            sim = self.ml.simulate(tmin=tmin, tmax=tmax)
            sim.plot(fig=fig)
        plt.xlim(tmin, tmax)
        plt.ylabel("Groundwater levels [meter]")
        plt.legend()

        return fig.axes

    @model_tmin_tmax
    def results(self, tmin=None, tmax=None, figsize=(10,8), **kwargs):
        """Plot different results in one window to get a quick overview.

        Parameters
        ----------
        tmin/tmax: str
            start and end time for plotting


        Returns
        -------

        """
        fig = self._get_figure(figsize=figsize)

        # Number of rows to make the figure with
        rows = 3 + len(self.ml.stressmodels)

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

        fig.tight_layout(pad=0.0)
        
        # Draw parameters table
        parameters = self.ml.parameters.copy()
        parameters['name']=parameters.index
        cols = ["name","optimal", "stderr"]
        parameters = parameters.loc[:, cols]
        for name, vals in parameters.loc[:, cols].iterrows():
            parameters.loc[name, "optimal"] = '{:.2f}'.format(vals.optimal)
            stderr_perc = np.abs(np.divide(vals.stderr, vals.optimal) * 100)
            parameters.loc[name, "stderr"] = '{:.1f}{}'.format(stderr_perc,
                                                             "\u0025")
        ax3.axis('off')
        # loc='upper center'
        ax3.table(bbox=(0.,0.,1.0,1.0), cellText=parameters.values,
                  colWidths=[0.5,0.25,0.25],colLabels=cols)

        return fig.axes

    @model_tmin_tmax
    def decomposition(self, tmin=None, tmax=None, ytick_base=True, split=True,
                      figsize=(10, 8), **kwargs):
        """Plot the decomposition of a time-series in the different stresses.

        Parameters
        ----------
        split
        ytick_base: Boolean or float
            Make the ytick-base constant if True, set this base to float if float
        **kwargs:
            Optional arguments for the subplots method

        Returns
        -------
        axes: list of matplotlib.axes

        """
        o = self.ml.observations(tmin=tmin, tmax=tmax)

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax)
        series = [sim]
        names = ['']

        # determine the influence of the different stresses
        for name in self.ml.stressmodels.keys():
            nstress = len(self.ml.stressmodels[name].stress)
            if split and nstress > 1:
                for istress in range(nstress):
                    contrib = self.ml.get_contribution(name, tmin=tmin,
                                                       tmax=tmax,
                                                       istress=istress)
                    series.append(contrib)
                    names.append(contrib.name)
            else:
                contrib = self.ml.get_contribution(name, tmin=tmin, tmax=tmax)
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

        # open the figure
        fig, ax = plt.subplots(len(series), sharex=True, figsize=figsize,
                               gridspec_kw={'height_ratios': height_ratios},
                               **kwargs)
        ax = np.atleast_1d(ax)

        # plot simulation and observations in top graph
        o_nu = self.ml.oseries.series.drop(o.index)
        if not o_nu.empty:
            # plot parts of the oseries that are not used in grey
            o_nu.plot(linestyle='', marker='.', color='0.5', label='',
                      markersize=2, ax=ax[0], x_compat=True)
        o.plot(linestyle='', marker='.', color='k',
               markersize=3, ax=ax[0], x_compat=True)
        sim.plot(ax=ax[0], x_compat=True)
        ax[0].set_title('Observations vs simulation')
        ax[0].set_ylim(ylims[0])
        ax[0].grid(which='both')
        ax[0].legend(ncol=3, frameon=False)


        if ytick_base:
            if isinstance(ytick_base, bool):
                # determine the ytick-spacing of the top graph
                yticks = ax[0].yaxis.get_ticklocs()
                if len(yticks) > 1:
                    ytick_base = yticks[1] - yticks[0]
                else:
                    ytick_base = None
            ax[0].yaxis.set_major_locator(
                MultipleLocator(base=ytick_base))

        # plot the influence of the stresses
        for i, contrib in enumerate(series[1:], start=1):
            contrib.plot(ax=ax[i], x_compat=True)

            if ytick_base:
                # set the ytick-spacing equal to the top graph
                ax[i].yaxis.set_major_locator(
                    MultipleLocator(base=ytick_base))

            ax[i].set_title(names[i])
            ax[i].set_ylim(ylims[i])
            ax[i].grid(which='both')
            ax[i].minorticks_off()

        ax[0].set_xlim(tmin, tmax)
        fig.tight_layout(pad=0.0)

        return ax

    @model_tmin_tmax
    def diagnostics(self, tmin=None, tmax=None):
        noise = self.ml.noise(tmin=tmin, tmax=tmax)

        fig = self._get_figure()
        gs = plt.GridSpec(2, 3, wspace=0.2)

        plt.subplot(gs[0, :2])
        plt.title('Autocorrelation')
        # plt.axhline(0.2, '--')
        r = acf(noise)
        plt.stem(r)

        plt.subplot(gs[1, :2])
        plt.title('Partial Autocorrelation')
        # plt.axhline(0.2, '--')
        # plt.stem(self.ml.stats.pacf())

        plt.subplot(gs[0, 2])
        noise.hist(bins=20)

        plt.subplot(gs[1, 2])
        probplot(noise, plot=plt)

        plt.xlim(tmin, tmax)

        return fig.axes

    def block_response(self, series=None):
        """Plot the block response for a specific series.

        Returns
        -------
        fig: matplotlib.axes
            matplotlib axes instance.

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
        # locs, labels = plt.xticks()
        # labels = locs * get_dt(self.ml.settings["freq"])
        # plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Block Response(s)")

        return fig.axes

    def step_response(self, series=None):
        """Plot the step response for a specific series.

        Returns
        -------
        axes: matplotlib.axes
            matplotlib axes instance.

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
        # locs, labels = plt.xticks()
        # labels = locs * get_dt(self.ml.settings["freq"])
        # plt.xticks(locs, labels)
        plt.xlabel("Time [days]")

        plt.legend(legend)
        fig.suptitle("Step Response(s)")

        return fig.axes

    @model_tmin_tmax
    def stresses(self, tmin=None, tmax=None, cols=1, split=True,
                 sharex=True, figsize=(10,8), **kwargs):
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

    def _get_figure(self, **kwargs):
        fig = plt.figure(**kwargs)
        return fig
