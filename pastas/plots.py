"""
This file contains the plotting functionalities that are available for Pastas.

Examples
--------
    ml.plot.decomposition()

"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from pandas import DataFrame, Timestamp
from scipy.stats import probplot

from .decorators import model_tmin_tmax
from .stats import acf

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

        Returns
        -------
        ax: matplotlib.axes
            matplotlib axes with the simulated and optionally the observed
            timeseries.

        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)

        ax.set_title("Results of {}".format(self.ml.name))

        if oseries:
            o = self.ml.observations()
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
        if legend:
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
        o = self.ml.observations()
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
            if len(title) > 3:
                title = title[:3] + ["..."]
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
        o = self.ml.observations()

        # determine the simulation
        sim = self.ml.simulate(tmin=tmin, tmax=tmax,
                               return_warmup=return_warmup)
        if name is not None:
            sim.name = name
        series = [sim]
        names = ['']

        # determine the influence of the different stresses
        for name in self.ml.stressmodels.keys():
            nsplit = self.ml.stressmodels[name].get_nsplit()
            if split and nsplit > 1:
                for istress in range(nsplit):
                    contrib = self.ml.get_contribution(
                        name, tmin=tmin, tmax=tmax, istress=istress,
                        return_warmup=return_warmup)
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
        if min_ylim_diff is not None:
            for i, ylim in enumerate(ylims):
                if np.diff(ylim) < min_ylim_diff:
                    ylims[i] = (np.mean(ylim) - min_ylim_diff / 2,
                                np.mean(ylim) + min_ylim_diff / 2)
        height_ratios = [
            0.0 if np.isnan(ylim[1] - ylim[0]) else ylim[1] - ylim[0] for ylim
            in ylims]

        if axes is None:
            # open a new figure
            gridspec_kw = {'height_ratios': height_ratios}
            fig, axes = plt.subplots(len(series), sharex=True, figsize=figsize,
                                     gridspec_kw=gridspec_kw, **kwargs)
            axes = np.atleast_1d(axes)
            o_label = o.name
            set_axes_properties = True
        else:
            if len(axes) != len(series):
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
    def diagnostics(self, tmin=None, tmax=None, figsize=(10, 8), **kwargs):
        """Plot a window that helps in diagnosing basic model assumptions.

        Parameters
        ----------
        tmin
        tmax

        Returns
        -------
        axes: list of matplotlib.axes

        """
        if self.ml.settings["noise"]:
            res = self.ml.noise(tmin=tmin, tmax=tmax)
        else:
            res = self.ml.residuals(tmin=tmin, tmax=tmax)

        fig = plt.figure(figsize=figsize, **kwargs)

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

        fig.tight_layout(pad=0.0)
        return fig.axes

    def block_response(self, stressmodels=None, ax=None, figsize=None,
                       **kwargs):
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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **kwargs)

        if not stressmodels:
            stressmodels = self.ml.stressmodels.keys()

        legend = []

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

    def step_response(self, stressmodels=None, ax=None, figsize=None,
                      **kwargs):
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
    def contributions_pie(self, tmin=None, tmax=None, ax=None,
                          figsize=None, **kwargs):
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
            _, ax = plt.subplots(figsize=figsize, **kwargs)

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
               wedgeprops=dict(width=1, edgecolor='w'))
        ax.axis('equal')
        return ax


class TrackSolve:
    """ Track and visualize optimization progress for pastas models.

    Parameters
    ----------
    ml : pastas.Model
        pastas Model to set up tracking for
    tmin : str or pandas.Timestamp, optional
        start time for simulation, by default None which
        defaults to first index in ml.oseries.series
    tmax : str or pandas.Timestamp, optional
        end time for simulation, by default None which
        defaults to last index in ml.oseries.series
    update_iter : int, optional
        update plot every update_iter iterations,
        by default 1

    Notes
    -----
    - Requires a matplotlib backend that supports interactive
      plotting, i.e. mpl.use("TkAgg").
    - Some possible speedups on the matplotlib side:
        - mpl.style.use("fast")
        - mpl.rcParams['path.simplify_threshold'] = 1.0
    - Since only parameters are passed to callback function in ml.solve,
      everything else passed to ml.solve must be known beforehand(?). This means
      if the tmin/tmax are passed in ml.solve() and not to TrackSolve(), the
      resulting plot will not correctly represent the statistics of the
      optimization.
    - TODO: check if more information passed to solve can be picked up
      from the model object instead of having to pass to TrackSolve.
    - TODO: check if statistics are calculated correctly as compared to
      results from ml.solve().
    - TODO: check if animation can be sped up somehow.
    - TODO: check what the relationship is between no. of iterations
      and the LeastSquares nfev and njev values. Model fit is only updated
      every few iterations ( = nparams?). Perhaps only update figure when
      fit and parameter values actually change?

    Examples
    --------
    Create a TrackSolve object for your model:

    >>> track = TrackSolve(ml)

    Initialize figure:

    >>> fig = track.initialize_figure()

    Solve model and pass track.update_figure as callback function:

    >>> ml.solve(callback=track.update_figure)

    """

    def __init__(self, ml, tmin=None, tmax=None, update_iter=1):
        logger.warning("TrackSolve feature under development. If you find any "
                       "bugs please comment on the issue on GitHub: "
                       "https://github.com/pastas/pastas/issues/137")

        self.ml = ml
        self.viewlim = 75  # no of iterations on axes by default
        self.update_iter = update_iter  # update plot every update_iter

        # get tmin/tmax
        if tmin is None:
            self.tmin = self.ml.oseries.series.index[0]
        else:
            self.tmin = Timestamp(tmin)

        if tmax is None:
            self.tmax = self.ml.oseries.series.index[-1]
        else:
            self.tmax = Timestamp(tmax)

        # parameters
        self.parameters = DataFrame(columns=self.ml.parameters.index)
        self.parameters.loc[0] = self.ml.parameters.initial.values

        # iteration counter
        self.itercount = 0

        # calculate RMSE residuals
        res = self._residuals(self.ml.parameters.initial.values)
        r_rmse = np.sqrt(np.sum(res ** 2))
        self.rmse_res = np.array([r_rmse])

        # calculate RMSE noise
        if self.ml.noisemodel is not None:
            noise = self._noise(self.ml.parameters.initial.values)
            n_rmse = np.sqrt(np.sum(noise ** 2))
            self.rmse_noise = np.array([n_rmse])

        # get observations
        self.obs = self.ml.observations(tmin=self.tmin,
                                        tmax=self.tmax)
        # calculate EVP
        self.evp = self._calc_evp(res.values, self.obs.values)

    def _append_params(self, params):
        """Append parameters to self.parameters DataFrame and
        update itercount, rmse values and evp.

        Parameters
        ----------
        params : np.array
            array containing parameters

        """
        # update itercount
        self.itercount += 1

        # add parameters to DataFrame
        self.parameters.loc[self.itercount,
                            self.ml.parameters.index] = params.copy()

        # calculate new RMSE values
        r_res = self._residuals(params)
        self.rmse_res = np.r_[self.rmse_res, np.sqrt(np.sum(r_res ** 2))]

        if self.ml.noisemodel is not None:
            n_res = self._noise(params)
            self.rmse_noise = np.r_[
                self.rmse_noise, np.sqrt(np.sum(n_res ** 2))]

        # recalculate EVP
        self.evp = self._calc_evp(r_res.values, self.obs.values)

    def _update_axes(self):
        """extend xlim if no. of iterations exceeds
        current window.

        """
        for iax in self.axes[1:]:
            iax.set_xlim(right=self.viewlim)
            self.fig.canvas.draw()

    def _update_settings(self):
        self.tmin = self.ml.settings["tmin"]
        self.tmax = self.ml.settings["tmax"]
        self.freq = self.ml.settings["freq"]

    def _calc_evp(self, res, obs):
        """ calculate evp
        """
        if obs.var() == 0.0:
            evp = 1.
        else:
            evp = max(0.0, (1 - (res.var(ddof=0) /
                                 obs.var(ddof=0))))
        return evp

    def _noise(self, params):
        """get noise

        Parameters
        ----------
        params : np.array
            array containing parameters

        Returns
        -------
        noise: np.array
            array containing noise

        """
        noise = self.ml.noise(parameters=params,
                              tmin=self.tmin,
                              tmax=self.tmax)
        return noise

    def _residuals(self, params):
        """calculate residuals

        Parameters
        ----------
        params : np.array
            array containing parameters

        Returns
        -------
        res: np.array
            array containing residuals

        """
        res = self.ml.residuals(parameters=params,
                                tmin=self.tmin,
                                tmax=self.tmax)
        return res

    def _simulate(self):
        """simulate model with last entry in self.parameters

        Returns
        -------
        sim: pd.Series
            series containing model evaluation

        """
        sim = self.ml.simulate(parameters=self.parameters.iloc[-1, :].values,
                               tmin=self.tmin,
                               tmax=self.tmax,
                               freq=self.ml.settings["freq"])
        return sim

    def initialize_figure(self, figsize=(10, 8), dpi=100):
        """Initialize figure for plotting optimization progress.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, passed to plt.subplots(), by default (10, 8)
        dpi : int, optional
            dpi of the figure passed to plt.subplots(), by default 100

        Returns
        -------
        fig: matplotlib.pyplot.Figure
            handle to the figure

        """
        # create plot
        self.fig, self.axes = plt.subplots(3, 1, figsize=(10, 8), dpi=100)
        self.ax0, self.ax1, self.ax2 = self.axes

        # plot oseries
        self.obs.plot(marker=".", ls="none", label="observations",
                      color="k", ms=4, x_compat=True, ax=self.ax0)

        # plot simulation
        sim = self._simulate()
        self.simplot, = self.ax0.plot(sim.index, sim, label="model")
        self.ax0.set_ylabel("oseries/model")
        self.ax0.set_title(
            "Iteration: {0} (EVP: {1:.2%})".format(self.itercount,
                                                   self.evp))
        self.ax0.legend(loc="lower right")

        # plot RMSE (residuals and/or residuals)
        legend_handles = []
        self.r_rmse_plot_line, = self.ax1.plot(
            range(self.itercount + 1), self.rmse_res, c="k", ls="solid",
            label="Residuals")
        self.r_rmse_plot_dot, = self.ax1.plot(
            self.itercount, self.rmse_res[-1], c="k", marker="o", ls="none")
        legend_handles.append(self.r_rmse_plot_line)
        self.ax1.set_xlim(0, self.viewlim)
        self.ax1.set_ylim(0, 1.05 * self.rmse_res[-1])
        self.ax1.set_ylabel("RMSE")

        if self.ml.noisemodel is not None:
            self.n_rmse_plot_line, = self.ax1.plot(
                range(self.itercount + 1), self.rmse_noise, c="C0", ls="solid",
                label="Noise")
            self.n_rmse_plot_dot, = self.ax1.plot(
                self.itercount, self.rmse_res[-1], c="C0", marker="o",
                ls="none")
            legend_handles.append(self.n_rmse_plot_line)
        legend_labels = [i.get_label() for i in legend_handles]
        self.ax1.legend(legend_handles, legend_labels, loc="upper right")

        # plot parameters values on semilogy
        plt.sca(self.ax2)
        plt.yscale("log")
        self.param_plot_handles = []
        legend_handles = []
        for pname, row in self.ml.parameters.iterrows():
            pa, = self.ax2.plot(
                range(self.itercount + 1), np.abs(row.initial), marker=".",
                ls="none", label=pname)
            pb, = self.ax2.plot(range(self.itercount + 1),
                                np.abs(row.initial), ls="solid",
                                c=pa.get_color())
            self.param_plot_handles.append((pa, pb))
            legend_handles.append(pa)

        legend_labels = [i.get_label() for i in legend_handles]
        self.ax2.legend(legend_handles, legend_labels, loc="lower right",
                        ncol=3)
        self.ax2.set_xlim(0, self.viewlim)
        self.ax2.set_ylim(1e-6, 1e5)
        self.ax2.set_ylabel("Parameter values")
        self.ax2.set_xlabel("Iteration")

        # set grid for each plot
        for iax in [self.ax0, self.ax1, self.ax2]:
            iax.grid(b=True)

        self.fig.tight_layout()
        return self.fig

    def update_figure(self, params):
        """Method to update figure while model is being solved. Pass this
        method to ml.solve(), e.g.:

        >>> track = TrackSolve(ml)
        >>> fig = track.initialize_figure()
        >>> ml.solve(callback=track.update_figure)

        Parameters
        ----------
        params : np.array
            array containing parameters

        """

        # update parameters
        self._append_params(params)

        # update settings from ml.settings
        self._update_settings()

        # check if figure should be updated
        if self.itercount % self.update_iter != 0:
            return

        # update view limits if needed
        if self.itercount >= self.viewlim:
            self.viewlim += 50
            self._update_axes()

        # update simulation
        sim = self._simulate()
        self.simplot.set_data(sim.index, sim.values)

        # update rmse residuals
        self.r_rmse_plot_line.set_data(
            range(self.itercount + 1), np.array(self.rmse_res))
        self.r_rmse_plot_dot.set_data(
            np.array([self.itercount]), np.array(self.rmse_res[-1]))
        # update rmse noise
        self.n_rmse_plot_line.set_data(
            range(self.itercount + 1), np.array(self.rmse_noise))
        self.n_rmse_plot_dot.set_data(
            np.array([self.itercount]), np.array(self.rmse_noise[-1]))

        # update parameter plots
        for j, (p1, p2) in enumerate(self.param_plot_handles):
            p1.set_data(np.array([self.itercount]),
                        np.abs(self.parameters.iloc[-1, j]))
            p2.set_data(range(self.itercount + 1),
                        self.parameters.iloc[:, j].abs().values)

        # update title
        self.ax0.set_title(
            "Iteration: {0} (EVP: {1:.2%})".format(self.itercount,
                                                   self.evp))
        self.fig.canvas.draw()
