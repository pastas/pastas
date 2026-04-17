"""Place to put TrackSolve"""

import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series, Timestamp

from pastas.stats import evp, rmse
from pastas.typing import ArrayLike, Model

logger = logging.getLogger(__name__)


class TrackSolve:
    """Track and/or visualize optimization progress for Pastas models.

    Parameters
    ----------
    ml : pastas.model.Model
        pastas Model to track
    tmin: pandas.Timestamp or str, optional
        start time for simulation, by default None which defaults to first index in
        ml.oseries.series
    tmax: pandas.Timestamp or str, optional
        end time for simulation, by default None which defaults to last index in
        ml.oseries.series
    update_iter : int, optional
        if visualizing optimization progress, update plot every update_iter
        iterations, by default nparam

    Notes
    -----
    Interactive plotting of optimization progress requires a matplotlib backend that
    supports interactive plotting, e.g. `mpl.use("TkAgg")` and `mpl.interactive(
    True)`. Some possible speedups on the matplotlib side include:
    - mpl.style.use("fast")
    - mpl.rcParams['path.simplify_threshold'] = 1.0

    Examples
    --------
    Set matplotlib backend and interactive mode (put this at the top of your script)::

        import matplotlib as mpl
        mpl.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()

    Create a TrackSolve object for your model::

        track = TrackSolve(ml)

    Solve model and store intermediate optimization results::

        ml.solve(callback=track.track_solve)

    Calculated parameters per iteration are stored in a pandas.DataFrame::

        track.parameters

    Other stored statistics include `track.evp` (explained variance percentage),
    `track.rmse_res` (root-mean-squared error of the residuals), `track.rmse_noise` (
    root mean squared error of the noise, only if a noisemodel is present).

    To interactively plot model optimization progress while solving pass
    `track.plot_track_solve` as callback function::

        ml.solve(callback=track.plot_track_solve)

    Access the resulting figure through `track.fig`.
    """

    def __init__(
        self,
        ml: Model,
        tmin: Timestamp | str | None = None,
        tmax: Timestamp | str | None = None,
        update_iter: int | None = None,
    ) -> None:
        logger.warning(
            "TrackSolve feature under development. If you find any bugs please post "
            "an issue on GitHub: https://github.com/pastas/pastas/issues"
        )

        self.ml = ml
        self.viewlim = 75  # no of iterations on axes by default
        if update_iter is None:
            self.update_iter = len(
                self.ml.parameters.loc[self.ml.parameters.vary].index
            )
        else:
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
        self.rmse_res = np.array([rmse(res=res)])

        # calculate RMSE noise
        if self.ml.noisemodel is not None:
            noise = self._noise(self.ml.parameters.initial.values)
            self.rmse_noise = np.array([rmse(res=noise)])

        # get observations
        self.obs = self.ml.observations(tmin=self.tmin, tmax=self.tmax)
        # calculate EVP
        self.evp = np.array([evp(obs=self.obs, res=res)])

    def track_solve(self, params: ArrayLike) -> None:
        """Append parameters to self.parameters DataFrame and update itercount,
        rmse values and evp.

        Parameters
        ----------
        params : array_like
            array containing parameters.
        """
        # update tmin/tmax and freq once after starting solve
        if self.itercount == 0:
            self._update_settings()

        # update itercount
        self.itercount += 1

        # add parameters to DataFrame
        self.parameters.loc[self.itercount, self.ml.parameters.index] = params.copy()

        # calculate new RMSE values
        r_res = self._residuals(params)
        self.rmse_res = np.r_[self.rmse_res, rmse(res=r_res)]

        if self.ml.noisemodel is not None:
            n_res = self._noise(params)
            self.rmse_noise = np.r_[self.rmse_noise, rmse(res=n_res)]

        # recalculate EVP
        self.evp = np.r_[self.evp, evp(obs=self.obs, res=r_res)]

    def _update_axes(self) -> None:
        """extend xlim if number of iterations exceeds current window."""
        for iax in self.axes[1:]:
            iax.set_xlim(right=self.viewlim)
            self.fig.canvas.draw()

    def _update_settings(self) -> None:
        self.tmin = self.ml.settings["tmin"]
        self.tmax = self.ml.settings["tmax"]
        self.freq = self.ml.settings["freq"]

    def _noise(self, params: ArrayLike) -> ArrayLike:
        """get noise.

        Parameters
        ----------
        params: array_like
            array containing parameters.

        Returns
        -------
        noise: array_like
            array containing noise.
        """
        noise = self.ml.noise(p=params, tmin=self.tmin, tmax=self.tmax)
        return noise

    def _residuals(self, params: ArrayLike) -> ArrayLike:
        """calculate residuals.

        Parameters
        ----------
        params: np.array
            array containing parameters.

        Returns
        -------
        res: array_like
            array containing residuals.
        """
        res = self.ml.residuals(p=params, tmin=self.tmin, tmax=self.tmax)
        return res

    def _simulate(self) -> Series:
        """simulate model with last entry in self.parameters.

        Returns
        -------
        sim: pd.Series
            Series containing model evaluation.
        """
        sim = self.ml.simulate(
            p=self.parameters.iloc[-1, :].values,
            tmin=self.tmin,
            tmax=self.tmax,
            freq=self.ml.settings["freq"],
        )
        return sim

    def initialize_figure(
        self, figsize: tuple[int] = (10, 8), dpi: int = 100
    ) -> Figure:
        """Initialize figure for plotting optimization progress.

        Parameters
        ----------
        figsize : tuple, optional
            figure size, passed to plt.subplots(), by default (10, 8).
        dpi : int, optional
            dpi of the figure passed to plt.subplots(), by default 100.

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            handle to the figure.
        """
        # create plot
        self.fig, self.axes = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
        self.ax0, self.ax1, self.ax2 = self.axes

        # share x-axes between 2nd and 3rd axes
        self.ax1.sharex(self.ax2)
        for t in self.ax1.get_xticklabels():
            t.set_visible(False)

        # plot oseries
        self.ax0.plot(
            self.obs.index,
            self.obs,
            marker=".",
            ls="none",
            label="observations",
            color="k",
            ms=4,
        )

        # plot simulation
        sim = self._simulate()
        (self.simplot,) = self.ax0.plot(sim.index, sim, label="simulation")
        self.ax0.set_ylabel("head")
        self.ax0.set_title(
            "Iteration: {0} (EVP: {1:.2f}%)".format(self.itercount, self.evp[-1])
        )
        self.ax0.legend(loc=(0, 1), frameon=False, ncol=2)
        omax = self.obs.max()
        omin = self.obs.min()
        vspace = 0.05 * (omax - omin)
        self.ax0.set_ylim(bottom=omin - vspace, top=omax + vspace)

        # plot RMSE (residuals and/or residuals)
        plt.sca(self.ax1)
        plt.yscale("log")
        legend_handles = []
        (self.r_rmse_plot_line,) = self.ax1.plot(
            [0], self.rmse_res[0:1], c="k", ls="solid", label="residuals"
        )
        (self.r_rmse_plot_dot,) = self.ax1.plot(
            self.itercount, self.rmse_res[-1], c="k", marker="o", ls="none"
        )
        legend_handles.append(self.r_rmse_plot_line)
        self.ax1.set_xlim(0, self.viewlim)
        self.ax1.set_ylim(1e-2, 2 * self.rmse_res[-1])
        self.ax1.set_ylabel("RMSE")

        if self.ml.noisemodel is not None:
            (self.n_rmse_plot_line,) = self.ax1.plot(
                [0], self.rmse_noise[0:1], c="C0", ls="solid", label="noise"
            )
            (self.n_rmse_plot_dot,) = self.ax1.plot(
                self.itercount, self.rmse_res[-1], c="C0", marker="o", ls="none"
            )
            legend_handles.append(self.n_rmse_plot_line)
        legend_labels = [i.get_label() for i in legend_handles]
        self.ax1.legend(
            legend_handles, legend_labels, loc=(0, 1), frameon=False, ncol=2
        )

        # plot parameters values on semilogy
        plt.sca(self.ax2)
        plt.yscale("log")
        self.param_plot_handles = []
        legend_handles = []
        for pname, row in self.ml.parameters.iterrows():
            if pname.startswith("noise"):
                if self.ml.noisemodel is None:
                    continue
            (pa,) = self.ax2.plot(
                [0], np.abs(row.initial), marker=".", ls="none", label=pname
            )
            (pb,) = self.ax2.plot(
                [0], np.abs(row.initial), ls="solid", c=pa.get_color()
            )
            self.param_plot_handles.append((pa, pb))
            legend_handles.append(pa)

        legend_labels = [i.get_label() for i in legend_handles]
        self.ax2.legend(
            legend_handles, legend_labels, loc=(0, 1), ncol=6, frameon=False
        )
        self.ax2.set_xlim(0, self.viewlim)
        self.ax2.set_ylim(1e-3, 1e4)
        self.ax2.set_ylabel("Parameter values")
        self.ax2.set_xlabel("Iteration")

        # set grid for each plot
        for iax in [self.ax0, self.ax1, self.ax2]:
            iax.grid(visible=True)

        self.fig.align_ylabels()
        self.fig.tight_layout()
        return self.fig

    def plot_track_solve(self, params: ArrayLike) -> None:
        """Method to plot model simulation while model is being solved.

        Parameters
        ----------
        params : array_like
            array containing parameters

        Examples
        --------
        Pass
        this method to ml.solve(), e.g.:

        >>> track = TrackSolve(ml)
        >>> ml.solve(callback=track.plot_track_solve)

        """
        if not hasattr(self, "fig"):
            self.initialize_figure()

        # update parameters
        self.track_solve(params)

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
            range(self.itercount + 1), np.array(self.rmse_res)
        )
        self.r_rmse_plot_dot.set_data(
            np.array([self.itercount]), np.array([self.rmse_res[-1]])
        )

        if self.ml.noisemodel is not None:
            # update rmse noise
            self.n_rmse_plot_line.set_data(
                range(self.itercount + 1), np.array(self.rmse_noise)
            )
            self.n_rmse_plot_dot.set_data(
                np.array([self.itercount]), np.array([self.rmse_noise[-1]])
            )

        # update parameter plots
        for j, (p1, p2) in enumerate(self.param_plot_handles):
            p1.set_data(
                np.array([self.itercount]), np.abs([self.parameters.iloc[-1, j]])
            )
            p2.set_data(
                range(self.itercount + 1), self.parameters.iloc[:, j].abs().values
            )

        # update title
        self.ax0.set_title(
            "Iteration: {0} (EVP: {1:.2f}%)".format(self.itercount, self.evp[-1])
        )
        plt.pause(1e-10)
        self.fig.canvas.draw()

    def plot_track_solve_history(self, fig: Figure | None = None) -> list[Axes]:
        """Plot optimization history.

        Parameters
        ----------
        fig : matplotlib.pyplot.Figure, optional
            figure handle, by default None, which constructs a new figure with
            `self.initialize_figure()`.

        Returns
        -------
        axes : list of matplotlib.pyplot.Axes
            list of axes handles in figure.
        """

        if fig is None:
            fig = self.initialize_figure()
        self.plot_track_solve(self.ml.parameters.optimal.values)

        self.fig.axes[1].autoscale(tight=False, axis="both")
        self.fig.axes[2].autoscale(tight=False, axis="both")

        self.fig.axes[1].set_xlim(left=0)
        # because of bug with autoscaling log axis?
        self.fig.axes[1].set_ylim(top=1.05 * self.rmse_res.max())

        return fig.axes
