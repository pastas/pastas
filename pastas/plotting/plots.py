"""This module contains plotting methods for Pastas."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series, Timestamp, concat
from scipy.stats import gaussian_kde, norm, pearsonr, probplot

from pastas.plotting.modelcompare import CompareModels
from pastas.plotting.plotutil import share_xaxes, share_yaxes
from pastas.stats.core import acf as get_acf
from pastas.stats.metrics import evp, rmse
from pastas.typing import ArrayLike, Axes, Figure, Model, TimestampType

logger = logging.getLogger(__name__)

__all__ = ["compare", "series", "acf", "diagnostics", "cum_frequency", "TrackSolve"]


def compare(
    models: List[Model],
    names: Optional[List[str]] = None,
    adjust_height: bool = True,
    **kwargs,
) -> Dict:
    """Plot multiple Pastas models in one figure to visually compare models.

    Notes
    -----
    The models must have the same stressmodel names, otherwise the contributions will
    not be plotted, and parameters table will not display nicely.

    Parameters
    ----------
    models: list
        List of pastas Models, works for N models, but certain things might not
        display nicely if the list gets too long.
    names : list of str
        override model names by passing a list of names
    adjust_height: bool, optional
        Adjust the height of the graphs, so that the vertical scale of all the
        subplots on the left is equal. Default is False, in which case the axes are
        not rescaled to include all data, so certain data might not be visible. Set
        False to ensure you can see all data.
    **kwargs
        The kwargs are passed to the CompareModels.plot() function.

    Returns
    -------
    matplotlib.axes
    """
    mc = CompareModels(models, names=names)
    mc.plot(adjust_height=adjust_height, **kwargs)
    return mc.axes


def series(
    head: Optional[Series] = None,
    stresses: Optional[List[Series]] = None,
    hist: bool = True,
    kde: bool = False,
    table: bool = False,
    titles: bool = True,
    tmin: Optional[TimestampType] = None,
    tmax: Optional[TimestampType] = None,
    labels: Optional[List[str]] = None,
    figsize: tuple = (10, 5),
) -> Axes:
    """Plot all the input time Series in a single plot.

    Parameters
    ----------
    head: pd.Series
        Pandas time series with DatetimeIndex.
    stresses: List of pd.Series
        List with Pandas time series with DatetimeIndex.
    hist: bool
        Histogram for the series. The number of bins is determined with Sturges rule.
    kde: bool
        Kernel density estimate for the series. The kde is obtained from
        scipy.gaussian_kde using scott to calculate the estimator bandwidth. Returns
        the number of observations, mean, skew and kurtosis.
    table: bool
        Show table with some basic statistics such as the number of
        observations, mean, skew and kurtosis.
    titles: bool
        Set the titles or not. Taken from the name attribute of the series.
    tmin: str or pd.Timestamp
    tmax: str or pd.Timestamp
    labels: List of str
        List with the labels for each subplot.
    figsize: tuple
        Set the size of the figure.

    Returns
    -------
    matplotlib.Axes
    """
    rows = 0
    if head is not None:
        rows += 1
        if tmin is None:
            tmin = head.index[0]
        if tmax is None:
            tmax = head.index[-1]
    if stresses is not None:
        rows += len(stresses)
    sharex = True
    gridspec_kw = {}
    cols = 1
    if table and not hist and kde:
        hist = True
    if hist or kde:
        sharex = False
        gridspec_kw["width_ratios"] = [3, 1]
        cols += 1
        if table:
            cols += 1
            gridspec_kw["width_ratios"].append(1)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        sharex=sharex,
        sharey="row",
        gridspec_kw=gridspec_kw,
    )
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    if hist:
        axes[-1, 1].set_xlabel("Frequency [%]")
    if kde:
        axes[-1, 1].set_xlabel("Density [-]")
    if head is not None:
        head = head[tmin:tmax].dropna()
        head.plot(ax=axes[0, 0], marker=".", linestyle=" ", color="k")
        if titles:
            axes[0, 0].set_title(head.name)
        if labels is not None:
            axes[0, 0].set_ylabel(labels[0])
        if hist and kde is False:
            head.hist(
                ax=axes[0, 1],
                orientation="horizontal",
                color="k",
                weights=np.ones(len(head)) / len(head) * 100,
                bins=int(np.ceil(1 + np.log2(len(head)))),
                grid=False,
            )
        if kde and hist:
            head.hist(
                ax=axes[0, 1],
                orientation="horizontal",
                color="k",
                bins=int(np.ceil(1 + np.log2(len(head)))),
                grid=False,
                density=True,
            )
        if kde:
            gkde = gaussian_kde(head, bw_method="scott")
            sample_range = np.max(head) - np.min(head)
            ind = np.linspace(
                np.min(head) - 0.1 * sample_range,
                np.max(head) + 0.1 * sample_range,
                1000,
            )
            if hist:
                colour = "C1"
            else:
                colour = "k"
            axes[0, 1].plot(gkde.evaluate(ind), ind, color=colour)
        if table:
            # stats table
            head_stats = [
                ["Count", f"{head.count():0.0f}"],
                ["Mean", f"{head.mean():0.2f}"],
                ["Max", f"{head.max():0.2f}"],
                ["Min", f"{head.min():0.2f}"],
                ["Skew", f"{head.skew():0.2f}"],
                ["Kurtosis", f"{head.kurtosis():0.2f}"],
            ]
            axes[0, 2].table(
                bbox=(0.0, 0.0, 1, 1), colWidths=(1.5, 1), cellText=head_stats
            )
            axes[0, 2].axis("off")

    if stresses is not None:
        for i, stress in enumerate(stresses, start=rows - len(stresses)):
            stress = stress[tmin:tmax].dropna()
            stress.plot(ax=axes[i, 0], color="k")
            if titles:
                axes[i, 0].set_title(stress.name)
            if labels is not None:
                axes[i, 0].set_ylabel(labels[i])
            if hist:
                # histogram
                stress.hist(
                    ax=axes[i, 1],
                    orientation="horizontal",
                    color="k",
                    weights=np.ones(len(stress)) / len(stress) * 100,
                    bins=int(np.ceil(1 + np.log2(len(stress)))),
                    grid=False,
                )
            if kde and hist:
                stress.hist(
                    ax=axes[i, 1],
                    orientation="horizontal",
                    color="k",
                    bins=int(np.ceil(1 + np.log2(len(stress)))),
                    grid=False,
                    density=True,
                )
            if kde:
                gkde = gaussian_kde(stress, bw_method="scott")
                sample_range = np.max(stress) - np.min(stress)
                ind = np.linspace(
                    np.min(stress) - 0.1 * sample_range,
                    np.min(stress) + 0.1 * sample_range,
                    1000,
                )
                if hist:
                    colour = "C1"
                else:
                    colour = "k"
                axes[i, 1].plot(gkde.evaluate(ind), ind, color=colour)
            if table:
                if i > 0:
                    axes[i, 0].sharex(axes[0, 0])
                # stats table
                stress_stats = [
                    ["Count", f"{stress.count():0.0f}"],
                    ["Mean", f"{stress.mean():0.2f}"],
                    ["Skew", f"{stress.skew():0.2f}"],
                    ["Kurtosis", f"{stress.kurtosis():0.2f}"],
                ]
                axes[i, 2].table(
                    bbox=(0, 0, 1, 1), colWidths=(1.5, 1), cellText=stress_stats
                )
                axes[i, 2].axis("off")

    # temporary fix, as set_xlim currently does not work with strings mpl=3.6.1
    if tmin is not None:
        tmin = Timestamp(tmin)
    if tmax is not None:
        tmax = Timestamp(tmax)
    axes[0, 0].set_xlim([tmin, tmax])
    axes[0, 0].minorticks_off()

    fig.tight_layout()
    return axes


def acf(
    series: Series,
    alpha: float = 0.05,
    lags: int = 365,
    acf_options: Optional[dict] = None,
    smooth_conf: bool = True,
    color: str = "k",
    ax: Optional[Axes] = None,
    figsize: tuple = (5, 2),
) -> Axes:
    """Plot of the autocorrelation function of a time series.

    Parameters
    ----------
    series: pandas.Series
        Residual series to plot the autocorrelation function for.
    alpha: float, optional
        Significance level to calculate the (1-alpha)-confidence intervals. For 95%
        confidence intervals, alpha should be 0.05.
    lags: int, optional
        Maximum number of lags (in days) to compute the autocorrelation for.
    acf_options: dict, optional
        Dictionary with keyword arguments passed on to pastas.stats.acf.
    smooth_conf: bool, optional
        For irregular time series the confidence interval may be.
    color: str, optional
        Color of the vertical autocorrelation lines.
    ax: matplotlib.axes.Axes, optional
        Matplotlib Axes instance to plot the ACF on. A new Figure and Axes is created
        when no value for ax is provided.
    figsize: Tuple, optional
        2-D Tuple to determine the size of the figure created. Ignored if ax is also
        provided.

    Returns
    -------
    ax: matplotlib.axes.Axes

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.rand(1000))
    >>> ps.plots.acf(res)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the autocorrelation
    if acf_options is None:
        acf_options = {}
    r = get_acf(series, full_output=True, alpha=alpha, lags=lags, **acf_options)

    if r.empty:
        raise ValueError(
            "The computed autocorrelation function has no values. Changing the input "
            "arguments ('acf_options') for calculating ACF may help."
        )

    if smooth_conf:
        conf = r.conf.rolling(10, min_periods=1).mean().values
    else:
        conf = r.conf.values

    ax.fill_between(r.index.days, conf, -conf, alpha=0.3)
    ax.vlines(r.index.days, [0], r.loc[:, "acf"].values, color=color)

    ax.set_xlabel("Lag [Days]")
    ax.set_xlim(0, r.index.days.max())
    ax.set_ylabel("Autocorrelation [-]")
    ax.set_title("Autocorrelation plot")

    ax.grid(True)
    return ax


def diagnostics(
    series: Series,
    sim: Optional[Series] = None,
    alpha: float = 0.05,
    bins: int = 50,
    acf_options: Optional[dict] = None,
    figsize: tuple = (10, 5),
    fig: Optional[Figure] = None,
    heteroscedasicity: bool = True,
    **kwargs,
) -> Axes:
    """Plot that helps in diagnosing basic model assumptions.

    Parameters
    ----------
    series: pandas.Series
        Pandas Series with the residual time series to diagnose.
    sim: pandas.Series, optional
        Pandas series with the simulated time series. Used to diagnose on
        heteroscedasticity. Ignored if heteroscedasticity is set to False.
    alpha: float, optional
        Significance level to calculate the (1-alpha)-confidence intervals.
    bins: int optional
        Number of bins used for the histogram. 50 is default.
    acf_options: dict, optional
        Dictionary with keyword arguments passed on to pastas.stats.acf.
    figsize: tuple, optional
        Tuple with the height and width of the figure in inches.
    fig: Matplotib.Figure instance, optional
        Optionally provide a Matplotib.Figure instance to plot onto.
    heteroscedasicity: bool, optional
        Create two additional subplots to check for heteroscedasticity. If true,
        a simulated time series has to be provided with the sim argument.
    **kwargs: dict, optional
        Optional keyword arguments, passed on to plt.figure.

    Returns
    -------
    axes: matplotlib.axes.Axes

    Examples
    --------
    >>> res = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.normal(0, 1, 1000))
    >>> ps.stats.plot_diagnostics(res)

    Notes
    -----
    The two right-hand side plots assume that the noise or residuals follow a Normal
    distribution.

    See Also
    --------
    pastas.stats.acf
        Method that computes the autocorrelation.
    scipy.stats.probplot
        Method use to plot the probability plot.
    """
    # Create the figure and axes
    if fig is None:
        fig = plt.figure(figsize=figsize, constrained_layout=True, **kwargs)

    if heteroscedasicity:
        if sim is None:
            msg = (
                "A simulated time series has to be provided to make plots to "
                "diagnose heteroscedasticity. Provide 'sim' argument."
            )
            logger.error(msg=msg)
            raise KeyError(msg)

        gs = fig.add_gridspec(ncols=3, nrows=2, width_ratios=[3, 1, 1])
        ax4 = fig.add_subplot(gs[0, 2])
        ax5 = fig.add_subplot(gs[1, 2])
    else:
        gs = fig.add_gridspec(ncols=2, nrows=2, width_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Plot the residuals or noise series
    ax.axhline(0, c="k")
    series.plot(ax=ax)
    ax.set_ylabel(series.name)
    ax.set_xlim(series.index.min(), series.index.max())
    ax.set_title(
        f"{series.name} (n={series.size :.0f}, $\\mu$" f"={series.mean() :.2f})"
    )
    ax.grid()
    ax.tick_params(axis="x", labelrotation=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")

    # Plot the autocorrelation
    acf(series, alpha=alpha, acf_options=acf_options, ax=ax1)
    ax1.set_title(None)

    # Plot the histogram for normality and add a 'best fit' line
    _, bins, _ = ax2.hist(series.values, bins=bins, density=True)
    y = norm.pdf(bins, series.mean(), series.std())
    ax2.plot(bins, y, "k--")
    ax2.set_ylabel("Probability density")
    ax2.set_title("Histogram")

    # Plot the probability plot
    _, (_, _, r) = probplot(series, plot=ax3, dist="norm", rvalue=False)
    c = ax.get_lines()[1].get_color()
    ax3.get_lines()[0].set_color(c)
    ax3.get_lines()[1].set_color("k")

    # Plot R2 here because probplot has suboptimal positioning
    ax3.text(0.5, 0.1, "$R^2={:.2f}$".format(r**2), transform=ax3.transAxes)

    if heteroscedasicity and sim is not None:
        # Plot residuals vs. simulation
        # interpolate simulation to times of observations
        sim = sim.loc[series.index]
        ax4.plot(sim, series, marker=".", linestyle=" ", color=c, alpha=0.7)
        ax4.grid()
        ax4.set_xlabel("Simulated values")
        ax4.set_ylabel("Residuals")

        # Plot residuals vs. simulation
        ax5.plot(
            sim, np.sqrt(series.abs()), marker=".", linestyle=" ", color=c, alpha=0.7
        )
        ax5.set_xlabel("Simulated values")
        ax5.set_ylabel("$\\sqrt{|Residuals|}$")
        ax5.grid()

    return fig.axes


def cum_frequency(
    obs: Series,
    sim: Optional[Series] = None,
    ax: Optional[Axes] = None,
    figsize: tuple = (5, 2),
) -> Axes:
    """Plot of the cumulative frequency of a time Series.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    obs: pandas.Series
        The pandas Series with the observed values.
    ax: matplotlib.axes.Axes, optional
        Matplotlib Axes instance to create the plot on. A new Figure and Axes is
        created when no value for ax is provided.
    figsize: Tuple, optional
        2-D Tuple to determine the size of the figure created. Ignored if ax is also
        provided.

    Returns
    -------
    ax: matplotlib.axes.Axes

    Examples
    --------
    >>> obs = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                 data=np.random.normal(0, 1, 1000))
    >>> ps.stats.plot_cum_frequency(obs)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(
        obs.sort_values(),
        np.arange(0, obs.size) / obs.size * 100,
        color="k",
        marker=".",
        linestyle=" ",
    )
    if sim is not None:
        ax.plot(sim.sort_values(), np.arange(0, sim.size) / sim.size * 100)
    ax.legend(["Observations", "Simulation"])
    ax.set_xlabel("Head")
    ax.set_ylabel("Cum. Frequency [%]")
    ax.grid()
    plt.tight_layout()

    return ax


class TrackSolve:
    """Track and/or visualize optimization progress for Pastas models.

    Parameters
    ----------
    ml : pastas.model.Model
        pastas Model to track
    tmin : str or pandas.Timestamp, optional
        start time for simulation, by default None which defaults to first index in
        ml.oseries.series
    tmax : str or pandas.Timestamp, optional
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
    root mean squared error of the noise, only if noise=True).

    To interactively plot model optimization progress while solving pass
    `track.plot_track_solve` as callback function::

        ml.solve(callback=track.plot_track_solve)

    Access the resulting figure through `track.fig`.
    """

    def __init__(
        self,
        ml: Model,
        tmin: Optional[TimestampType] = None,
        tmax: Optional[TimestampType] = None,
        update_iter: Optional[int] = None,
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
        if self.ml.settings["noise"] and self.ml.noisemodel is not None:
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

        if self.ml.settings["noise"] and self.ml.noisemodel is not None:
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
        self, figsize: Tuple[int] = (10, 8), dpi: int = 100
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

        if self.ml.settings["noise"] and self.ml.noisemodel is not None:
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
                if not self.ml.settings["noise"] or self.ml.noisemodel is None:
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

        if self.ml.settings["noise"] and self.ml.noisemodel is not None:
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

    def plot_track_solve_history(self, fig: Optional[Figure] = None) -> List[Axes]:
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


def pairplot(
    data: Union[DataFrame, List[Series]],
    bins: Optional[int] = None,
) -> Dict[str, Axes]:
    """Plot correlation between time series on of values on the same time steps.
    Based on seaborn pairplot method.
    Parameters
    ----------
    data : Union[DataFrame, List[Series]]
        List of Series or Dataframe with DateTime index
    bins : Optional[int], optional
        Number of bins in the histogram, by default None which uses Sturge's
        Rule to determine the number bins
    Returns
    -------
    Dict[str, Axes]
    """
    if isinstance(data, list):
        data = concat(data, axis=1)

    df = data.dropna(how="any")

    columns = df.columns

    mosaic = []
    for i, column in enumerate(columns):
        cols = [f"scatter_{x}-{column}" for x in columns]
        cols[i] = f"hist_{column}"
        mosaic.append(cols)

    mosaic = np.array(mosaic)

    f, axd = plt.subplot_mosaic(mosaic, figsize=(6.5, 6))

    for i, (column, mos) in enumerate(zip(columns, mosaic)):
        # plot histogram
        if bins is None:
            bins = int(np.ceil(1 + np.log2(len(df.loc[:, column].values))))
        counts, bins = np.histogram(df.loc[:, column].values, bins=bins)
        scaled_counts = (
            df.loc[:, column].max()
            * (counts - np.min(counts))
            / (np.max(counts) - np.min(counts))
        )
        axd[f"hist_{column}"].hist(x=bins[:-1], bins=bins, weights=scaled_counts)

        # plot scatter
        other_cols = [x for x in columns if x is not column]
        for col in other_cols:
            axd[f"scatter_{column}-{col}"].scatter(
                df.loc[:, column].values,
                df.loc[:, col].values,
                alpha=0.6,
                s=20,
                edgecolor="white",
                linewidth=0.3,
            )
            r, _ = pearsonr(df.loc[:, column].values, df.loc[:, col].values)
            axd[f"scatter_{column}-{col}"].annotate(
                f"r = {r:.2f}",
                xy=(0.5, 0.95),
                horizontalalignment="center",
                verticalalignment="top",
                xycoords="axes fraction",
                color="k",
                path_effects=[
                    path_effects.withStroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ],
            )

        # set labels
        axd[mos[0]].set_ylabel(column)
        if (mos == mosaic[-1]).all():
            _ = [axd[j].set_xlabel(x) for x, j in zip(columns, mos)]

        # share x and y axis per row and columns
        share_yaxes([axd[j] for j in mos])

    _ = [share_xaxes([axd[x] for x in mosaic[:, j]]) for j in range(len(columns))]

    f.tight_layout()

    return axd
