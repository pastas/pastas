"""This module contains plotting methods for Pastas."""

import logging

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series, Timestamp, concat
from scipy.stats import gaussian_kde, norm, pearsonr, probplot

from pastas.decorators import PastasDeprecationWarning
from pastas.plotting.modelcompare import CompareModels
from pastas.plotting.plotutil import plot_series_with_gaps, share_xaxes, share_yaxes
from pastas.stats.core import acf as get_acf
from pastas.typing import Axes, Figure, Model

logger = logging.getLogger(__name__)

__all__ = ["compare", "series", "acf", "diagnostics", "cum_frequency"]


@PastasDeprecationWarning(
    version="2.0.0",
    reason="The TrackSolve class has been moved to pastas.solver.trackers.TrackSolve.",
)
class TrackSolve:
    pass


def compare(
    models: list[Model],
    names: list[str] | None = None,
    adjust_height: bool = True,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
    **kwargs,
) -> dict:
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
    tmin: pandas.Timestamp or str, optional
        A string or pandas.Timestamp with the start date for the
        simulation period (E.g. '1980-01-01 00:00:00'). If none
        is provided, the tmin from the oseries is used.
    tmax: pandas.Timestamp or str, optional
        A string or pandas.Timestamp with the end date for the
        simulation period (E.g. '2020-01-01 00:00:00'). If none
        is provided, the tmax from the oseries is used.
    **kwargs
        The kwargs are passed to the CompareModels.plot() function.

    Returns
    -------
    matplotlib.axes
    """
    mc = CompareModels(models, names=names, tmin=tmin, tmax=tmax)
    mc.plot(adjust_height=adjust_height, **kwargs)
    return mc.axes


def series(
    head: Series | None = None,
    stresses: list[Series] | None = None,
    hist: bool = True,
    kde: bool = False,
    table: bool = False,
    titles: bool = True,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
    colors_stresses: list[str] | None = None,
    labels: list[str] | None = None,
    figsize: tuple = (10, 5),
    **kwargs,
) -> Axes:
    """Plot all the input time Series in a single plot.

    Parameters
    ----------
    head: pd.Series
        Pandas time series with DatetimeIndex.
    stresses: list of pd.Series
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
    tmin: str or Timestamp
    tmax: str or Timestamp
    colors_stresses: list of str
        List with the matplotlib colorcodes to use for plotting each stress timeseries.
        If list is shorter than number of stresses, the remaining stresses are plotted
        in black. If None (default), default matplotlib colors will be used.
    labels: list of str
        List with the labels for each subplot.
    figsize: tuple
        Set the size of the figure.
    kwargs:
        keyword arguments passed to plotting functions of stress timeseries

    Returns
    -------
    matplotlib.Axes
    """
    nrows = 0
    if head is not None:
        nrows += 1
        tmin = head.index[0] if tmin is None else tmin
        tmax = head.index[-1] if tmax is None else tmax
    if stresses is not None:
        nrows += len(stresses)
    if colors_stresses is None:
        colors_stresses = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    gridspec_kw = {}
    cols = 1
    if table and not hist and not kde:
        logging.info(
            "Plotting the table is not possible without hist=True or kde=True. Adding the histogram."
        )
        hist = True
    if hist or kde:
        gridspec_kw["width_ratios"] = [3, 1]
        cols += 1
        if table:
            cols += 1
            gridspec_kw["width_ratios"].append(1)
    _, axes = plt.subplots(
        nrows,
        cols,
        figsize=figsize,
        sharey="row",
        gridspec_kw=gridspec_kw,
    )
    if nrows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis]
    elif cols == 1:
        axes = axes[:, np.newaxis]
    if hist:
        axes[-1, 1].set_xlabel("Frequency [%]")
    if kde:
        axes[-1, 1].set_xlabel("Density [-]")
    if head is not None:
        head = head.loc[tmin:tmax].dropna()
        head.plot(
            ax=axes[0, 0], marker=".", linestyle=" ", color="k", xlabel="", **kwargs
        )
        if titles:
            axes[0, 0].set_title(head.name)
        if labels is not None:
            axes[0, 0].set_ylabel(labels[0])
        if hist:
            weights = None if kde else np.ones(len(head)) / len(head) * 100
            head.hist(
                ax=axes[0, 1],
                orientation="horizontal",
                color="k",
                weights=weights,
                bins=int(np.ceil(1 + np.log2(len(head)))),
                grid=False,
                density=kde,
            )
        if kde:
            gkde = gaussian_kde(head, bw_method="scott")
            sample_range = np.max(head) - np.min(head)
            ind = np.linspace(
                np.min(head) - 0.1 * sample_range,
                np.max(head) + 0.1 * sample_range,
                1000,
            )
            color = "darkgrey" if hist else "k"
            axes[0, 1].plot(gkde.evaluate(ind), ind, color=color)
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
        for i, stress in enumerate(stresses, start=nrows - len(stresses)):
            stress = stress.loc[tmin:tmax].dropna()
            if i <= len(colors_stresses):
                color_stress = colors_stresses[i - 1]
            else:
                color_stress = "k"
            stress.plot(ax=axes[i, 0], color=color_stress, xlabel="", **kwargs)
            if titles:
                axes[i, 0].set_title(stress.name)
            if labels is not None:
                axes[i, 0].set_ylabel(labels[i])
            if hist:
                weights = None if kde else np.ones(len(stress)) / len(stress) * 100
                stress.hist(
                    ax=axes[i, 1],
                    orientation="horizontal",
                    color=color_stress,
                    weights=weights,
                    bins=int(np.ceil(1 + np.log2(len(stress)))),
                    grid=False,
                    density=kde,
                )
            if kde:
                gkde = gaussian_kde(stress, bw_method="scott")
                sample_range = np.max(stress) - np.min(stress)
                ind = np.linspace(
                    np.min(stress) - 0.1 * sample_range,
                    np.max(stress) + 0.1 * sample_range,
                    1000,
                )
                color = "darkgrey" if hist else color_stress
                axes[i, 1].plot(gkde.evaluate(ind), ind, color=color)
            if table:
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

    share_xaxes(axes[:, 0])

    return axes


def acf(
    series: Series,
    alpha: float = 0.05,
    lags: int = 365,
    acf_options: dict | None = None,
    smooth_conf: bool = True,
    color: str = "k",
    ax: Axes | None = None,
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
    figsize: tuple, optional
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

    Raises
    ------
    Warning if the ACF is empty. The plot will still be created to ensure that scripts
    will still run when dealing with many models.

    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the autocorrelation
    if acf_options is None:
        acf_options = {}
    r = get_acf(series, full_output=True, alpha=alpha, lags=lags, **acf_options)

    if r.empty:
        # Raise a warning
        logger.warning(
            "The computed autocorrelation function has no values. Changing the input "
            "arguments ('acf_options') for calculating ACF may help. No data will be "
            "plotted."
        )
    else:
        if smooth_conf:
            conf = r.conf.rolling(10, min_periods=1).mean().values
        else:
            conf = r.conf.values

        ax.fill_between(
            r.index.days,
            conf,
            -conf,
            alpha=0.3,
            label=f"{1 - alpha:.0%} confidence interval",
        )
        ax.legend(loc="lower right")
        ax.vlines(r.index.days, [0], r.loc[:, "acf"].values, color=color)
        ax.set_xlim(0, r.index.days.max())

    ax.set_xlabel("Lag [Days]")
    ax.set_ylabel("Autocorrelation [-]")
    ax.set_title("Autocorrelation plot")

    ax.grid(True)
    return ax


def diagnostics(
    series: Series,
    sim: Series | None = None,
    alpha: float = 0.05,
    bins: int = 50,
    acf_options: dict | None = None,
    figsize: tuple = (10, 5),
    fig: Figure | None = None,
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
    ax = plot_series_with_gaps(series, ax=ax)
    ax.set_ylabel(series.name)
    ax.set_xlim(series.index.min(), series.index.max())
    ax.set_title(f"{series.name} (n={series.size:.0f}, $\\mu$={series.mean():.2f})")
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
    sim: Series | None = None,
    ax: Axes | None = None,
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
    figsize: tuple, optional
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


def pairplot(
    data: DataFrame | list[Series],
    bins: int | None = None,
) -> dict[str, Axes]:
    """Plot correlation between time series on of values on the same time steps.

    Based on seaborn pairplot method.

    Parameters
    ----------
    data : DataFrame | list[Series]
        List of Series or Dataframe with DateTime index
    bins : int | None, optional
        Number of bins in the histogram, by default None which uses Sturge's
        Rule to determine the number bins

    Returns
    -------
    dict[str, Axes]
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
