"""Module containing plotting methods for Pastas."""

import logging

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series, Timedelta, Timestamp, concat
from scipy.stats import gaussian_kde, norm, pearsonr, probplot

from pastas.decorators import PastasDeprecationWarning, deprecate_args_or_kwargs
from pastas.plotting.modelcompare import CompareModels
from pastas.plotting.plotutil import plot_series_with_gaps, share_xaxes, share_yaxes
from pastas.stats.core import acf as get_acf
from pastas.typing import Axes, Model

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
) -> dict[str, Axes]:
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
        dict[str, matplotlib.axes.Axes]
    """
    mc = CompareModels(models, names=names, tmin=tmin, tmax=tmax)
    mc.plot(adjust_height=adjust_height, **kwargs)
    return mc.axes


def series(
    oseries: Series | None = None,
    stresses: list[Series] | None = None,
    hist: bool = True,
    kde: bool = False,
    table: bool = False,
    titles: bool = True,
    tmin: Timestamp | str | None = None,
    tmax: Timestamp | str | None = None,
    colors_stresses: list[str] | None = None,
    labels: list[str] | None = None,
    figsize: tuple = (8.0, 4.0),
    **kwargs,
) -> Axes:
    """Plot all the input time Series in a single plot.

    Parameters
    ----------
    oseries: pd.Series
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
    if "head" in kwargs:
        deprecate_args_or_kwargs(
            name="head",
            version="2.3.0",
            reason="Please use `oseries` instead of `head`.",
        )
        if oseries is None:
            oseries = kwargs.pop("head")
        else:
            kwargs.pop("head")

    nrows = 0
    if oseries is not None:
        nrows += 1
        tmin = oseries.index[0] if tmin is None else tmin
        tmax = oseries.index[-1] if tmax is None else tmax
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
    if oseries is not None:
        oseries = oseries.loc[tmin:tmax].dropna()
        oseries.plot(
            ax=axes[0, 0], marker=".", linestyle=" ", color="k", xlabel="", **kwargs
        )
        if titles:
            axes[0, 0].set_title(oseries.name)
        if labels is not None:
            axes[0, 0].set_ylabel(labels[0])
        if hist:
            weights = None if kde else np.ones(len(oseries)) / len(oseries) * 100
            oseries.hist(
                ax=axes[0, 1],
                orientation="horizontal",
                color="k",
                weights=weights,
                bins=int(np.ceil(1 + np.log2(len(oseries)))),
                grid=False,
                density=kde,
            )
        if kde:
            gkde = gaussian_kde(oseries, bw_method="scott")
            sample_range = np.max(oseries) - np.min(oseries)
            ind = np.linspace(
                np.min(oseries) - 0.1 * sample_range,
                np.max(oseries) + 0.1 * sample_range,
                1000,
            )
            color = "darkgrey" if hist else "k"
            axes[0, 1].plot(gkde.evaluate(ind), ind, color=color)
        if table:
            # stats table
            oseries_stats = [
                ["Count", f"{oseries.count():0.0f}"],
                ["Mean", f"{oseries.mean():0.2f}"],
                ["Max", f"{oseries.max():0.2f}"],
                ["Min", f"{oseries.min():0.2f}"],
                ["Skew", f"{oseries.skew():0.2f}"],
                ["Kurtosis", f"{oseries.kurtosis():0.2f}"],
            ]
            axes[0, 2].table(
                bbox=(0.0, 0.0, 1, 1), colWidths=(1.5, 1), cellText=oseries_stats
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
    **kwargs,
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
    **kwargs: dict, optional
        Optional keyword arguments, passed on to plt.subplots.

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
    kwargs = {} or kwargs
    if ax is None:
        figsize = kwargs.pop("figsize", (5.0, 3.0))
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
    heteroscedasicity: bool = True,
    max_plot_gap: Timedelta | float | None = None,
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
    heteroscedasicity: bool, optional
        Create two additional subplots to check for heteroscedasticity. If true,
        a simulated time series has to be provided with the sim argument.
    max_plot_gap: Timedelta | float | None, optional
        Maximum gap to be considered as a gap. If the difference between two
        consecutive index values is larger than max_plot_gap, a gap is inserted in the
        plot. Default is None.
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
    if heteroscedasicity and sim is None:
        msg = (
            "A simulated time series has to be provided to make plots to "
            "diagnose heteroscedasticity. Provide 'sim' argument."
        )
        logger.error(msg=msg)
        raise KeyError(msg)

    # Create the figure and axes
    kwargs = {} or kwargs
    figsize = kwargs.pop("figsize", (8.0, 4.0))
    layout = kwargs.pop("layout", "constrained")
    if heteroscedasicity:
        mosaic = [["series", "hist", "het_res"], ["acf", "qq", "het_sqrt"]]
        width_ratios = kwargs.pop("width_ratios", [3, 1, 1])
    else:
        mosaic = [["series", "hist"], ["acf", "qq"]]
        width_ratios = kwargs.pop("width_ratios", [3, 1])

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig, axd = plt.subplot_mosaic(
            mosaic=mosaic,
            figsize=figsize,
            width_ratios=width_ratios,
            layout=layout,
            **kwargs,
        )
    else:
        axd = fig.subplot_mosaic(
            mosaic=mosaic,
            width_ratios=width_ratios,
            **kwargs,
        )

    # Plot the residuals or noise series
    axd["series"].axhline(0, c="k")
    axd["series"] = plot_series_with_gaps(series, ax=axd["series"], gap=max_plot_gap)
    axd["series"].set_ylabel(series.name)
    axd["series"].set_xlim(series.index.min(), series.index.max())
    axd["series"].set_title(
        f"{series.name} (n={series.size:.0f}, $\\mu$={series.mean():.2f})"
    )
    axd["series"].grid(True)
    axd["series"].tick_params(axis="x", labelrotation=0)
    for label in axd["series"].get_xticklabels():
        label.set_horizontalalignment("center")

    # Plot the autocorrelation
    acf(series, alpha=alpha, acf_options=acf_options, ax=axd["acf"])
    axd["acf"].set_title(None)

    # Plot the histogram for normality and add a 'best fit' line
    _, bins, _ = axd["hist"].hist(series.values, bins=bins, density=True)
    y = norm.pdf(bins, series.mean(), series.std())
    axd["hist"].plot(bins, y, "k--")
    axd["hist"].set_ylabel("Probability density")
    axd["hist"].set_title("Histogram")

    # Plot the probability plot
    _, (_, _, r) = probplot(series, plot=axd["qq"], dist="norm", rvalue=False)
    c = axd["series"].get_lines()[1].get_color()
    axd["qq"].get_lines()[0].set_color(c)
    axd["qq"].get_lines()[1].set_color("k")

    # Plot R2 here because probplot has suboptimal positioning
    axd["qq"].text(0.5, 0.1, "$R^2={:.2f}$".format(r**2), transform=axd["qq"].transAxes)

    if heteroscedasicity and sim is not None:
        # Plot residuals vs. simulation
        # interpolate simulation to times of observations
        sim = sim.loc[series.index]
        axd["het_res"].plot(sim, series, marker=".", linestyle=" ", color=c, alpha=0.7)
        axd["het_res"].grid(True)
        axd["het_res"].set_xlabel("Simulated values")
        axd["het_res"].set_ylabel("Residuals")

        # Plot residuals vs. simulation
        axd["het_sqrt"].plot(
            sim, np.sqrt(series.abs()), marker=".", linestyle=" ", color=c, alpha=0.7
        )
        axd["het_sqrt"].set_xlabel("Simulated values")
        axd["het_sqrt"].set_ylabel("$\\sqrt{|Residuals|}$")
        axd["het_sqrt"].grid(True)

    return fig.axes


def cum_frequency(
    oseries: Series | None = None,
    sim: Series | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Plot of the cumulative frequency of a time Series.

    Parameters
    ----------
    sim: pandas.Series
        Series with the simulated values.
    oseries: pandas.Series
        The pandas Series with the observed values.
    ax: matplotlib.axes.Axes, optional
        Matplotlib Axes instance to create the plot on. A new Figure and Axes is
        created when no value for ax is provided.
    **kwargs: dict, optional
        Optional keyword arguments, passed on to plt.subplots.

    Returns
    -------
    ax: matplotlib.axes.Axes

    Examples
    --------
    >>> oseries = pd.Series(index=pd.date_range(start=0, periods=1000, freq="D"),
    >>>                     data=np.random.normal(0, 1, 1000))
    >>> ps.stats.plot_cum_frequency(oseries)
    """
    if "obs" in kwargs:
        deprecate_args_or_kwargs(
            name="obs",
            version="2.3.0",
            reason="Please use `oseries` instead of `obs`.",
        )
        if oseries is None:
            oseries = kwargs.pop("obs")
        else:
            kwargs.pop("obs")

    if oseries is None:
        raise TypeError("cum_frequency() missing required argument: 'oseries'")

    kwargs = {} or kwargs
    if ax is None:
        figsize = kwargs.pop("figsize", (5.0, 3.0))
        _, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

    ax.plot(
        oseries.sort_values(),
        np.arange(0, oseries.size) / oseries.size * 100,
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
