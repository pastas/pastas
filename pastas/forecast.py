from logging import getLogger

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from numpy import linspace, ones
from numpy.random import randn
from pandas import DataFrame, MultiIndex, Timedelta, concat, date_range

logger = getLogger(__name__)


def _check_forecast_data(forecasts):
    """Internal method to check the integrity of the forecasts data.

    Parameters
    ----------
    forecasts: dict
        Dictionary containing the forecasts data. The keys are the stressmodel names
        and the values are lists of DataFrames containing the forecasts with a datetime
        index and each column a time series (i.e., one ensemble member).

    Returns
    -------
    n: int
        The number of ensemble members in the forecasts.
    tmin: datetime
        The minimum datetime in the forecasts.
    tmax: datetime
        The maximum datetime in the forecasts.
    index: DatetimeIndex
        The datetime index of the forecasts.

    Notes
    -----
    This method checks if the number of columns and indices are the same for all
    DataFrames in the forecasts dictionary. If the number of columns is not the same,
    a warning is printed and a ValueError is raised. If the indices are not the same,
    a warning is printed and a ValueError is raised.

    """
    n = None
    tmax = None
    tmin = None
    index = None

    for fc_data in forecasts.values():
        for fc in fc_data:
            # Check if the number of columns is the same for all DataFrames
            if n is None:
                n = fc.columns.size
                tmin = fc.index[0]
                tmax = fc.index[-1]
                index = fc.index
            # If the number of columns is not the same, print a warning
            elif n != fc.columns.size:
                msg = (
                    "The number of ensemble members is not the same for all forecasts. "
                    "Please check the forecast data."
                )
                logger.error(msg)
                raise ValueError(msg)
            elif tmin != fc.index[0] or tmax != fc.index[-1]:
                msg = (
                    "The time index of the forecasts is not the same for all forecasts."
                    "tmax Please check the forecast data."
                )
                logger.error(msg)
                raise ValueError(msg)
            else:
                pass

    return n, tmin, tmax, index


def forecast(ml, forecasts, nparam, post_process=False):
    """Method to forecast the head from ensembles of stress forecasts.

    Parameters
    ----------
    ml: pastas.Model
        Pastas Model instance.
    forecasts: dict
        Dictionary containing the forecasts data. The keys are the stressmodel names
        and the values are lists of DataFrames containing the forecasts with a datetime
        index and each column a time series (i.e., one ensemble member).
    nparam: int
        The number of parameters to generate.
    post_process: bool, optional
        If True, the forecasts are post-processed using the noise model of the model
        instance. Default is False. If True, a noise model should be present in the
        model instance. If not, an error is raised.

    Returns
    -------
    nopp: pandas.DataFrame
        DataFrame containing the forecasts without post-processing. The columns are
        a MultiIndex with the first level the ensemble member and the second level the
        parameter member.
    pp: pandas.DataFrame
        DataFrame containing the forecasts with post-processing. The columns are
        a MultiIndex with the first level the ensemble member and the second level the
        parameter member. Only returned if post_process is True.

    Notes
    -----


    """
    # Copy the model so old model is unaffected.
    ml2 = ml.copy()

    # Check the integrity of the forecasts data
    n, tmin, tmax, index = _check_forecast_data(forecasts)

    # Get the residuals
    res = ml2.residuals(tmax=tmin - Timedelta(1, "D")).dropna()

    # Create DataFrames to store data
    mi = MultiIndex.from_product(
        [range(n), range(nparam)], names=["meteo_member", "param_member"]
    )
    nopp = DataFrame(index=index, columns=mi, dtype=float)

    if post_process:
        if ml.noisemodel is None:
            msg = (
                "No noisemodel is present in the model instance. "
                "Please add a noisemodel to the model instance."
            )
            logger.error(msg)
            raise ValueError(msg)

        sigr = ml2.noise().std()
        alpha = ml2.parameters.loc["noise_alpha", "optimal"]

        pp = DataFrame(index=index, columns=mi, dtype=float)
        correction = (
            ml2.noisemodel.get_correction(res, [alpha], index).values
            * ones((nparam, 1))
        ).T

    # Generate forecasts with each ensemble member
    for m in range(n):
        # Update stresses with ensemble member data
        for sm_name, fc_data in forecasts.items():
            # Select stressmodel
            sm = ml2.stressmodels[sm_name]

            # Update stress with forecast data from a single member
            for i, fc in enumerate(fc_data):
                ts = concat(
                    [
                        sm.stress[i].series_original.loc[: tmin - Timedelta(1, "D")],
                        fc.iloc[:, m],
                    ]
                )
                sm.stress[i].series_original = ts

        # Generate
        sim = ml2.simulate(tmin=tmin, tmax=tmax)
        raw = (sim.values * ones((nparam, 1))).T
        raw = raw + sigr * randn(sim.index.size, nparam)

        nopp.loc[:, (m, slice(None))] = raw
        pp.loc[:, (m, slice(None))] = raw + correction

    if post_process:
        return nopp, pp
    else:
        nopp


def plot_forecast(
    head,
    sim=None,
    forecasts=None,
    tmin=None,
    tmax=None,
    ax=None,
    plot_history=False,
    days_before=5,
    quantiles=None,
    cmap="RdYlBu",
    title=None,
    legend=True,
):
    """Method to plot results of groundwater forecasts.

    Parameters
    ----------
    head
    sim
    forecasts
    tmin
    tmax
    ax
    year
    days_before
    quantiles
    cmap

    Returns
    -------


    """

    # Create matplotlib axes if none is provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.figure

    tmin = forecasts.index[0] - Timedelta("30D")

    # Plot the forecasts ensembles if forecasts are provided
    if forecasts is not None:
        med = forecasts.median(axis=1).plot(ax=ax, zorder=11, color="C1")
        tmax = forecasts.index[-1]
        forecasts.T.groupby(level=0).quantile(0.5).T.plot(
            color="C0", legend=False, ax=ax, alpha=0.7
        )

        df = forecasts.T.quantile([0, 1]).T
        prob = ax.fill_between(
            df.index, df.values[:, 0], df.values[:, 1], alpha=0.6, color="gray"
        )

    if plot_history:
        # Plot the quantiles in the background
        if quantiles is None:
            quantiles = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]

            cmap_colors = cmap(linspace(0.1, 0.9, len(quantiles) - 1))

            fills = []
            group = head.rolling(Timedelta("14D")).mean().groupby(head.index.dayofyear)

            index = date_range(tmin, tmax, freq="D")

            for i in range(5):
                l = group.quantile(quantiles[i])[index.dayofyear]
                u = group.quantile(quantiles[i + 1])[index.dayofyear]
                fill = ax.fill_between(
                    index, l.values, u.values, color=cmap_colors[i], alpha=0.5
                )
                fills.append(fill)

    head.loc[tmin:tmax].plot(
        marker=".", color="k", linestyle=" ", markersize=3, ax=ax, zorder=10
    )

    ax.axvline(forecasts.index[0], color="k", linestyle="--")
    ax.grid()
    ax.set_xlim(tmin, forecasts.index[-1])
    ax.set_ylabel("Groundwater Level [m]")

    ll, bb, ww, hh = ax.get_position().bounds

    if legend:
        ho = mlines.Line2D([], [], color="k", marker=".", linestyle=" ")
        med = mlines.Line2D([], [], color="C1")
        ens = mlines.Line2D([], [], color="C0")
        ax.legend(
            [ho, med, ens, prob],
            ["Observed", "Median", "51 member", "probab. range"],
            numpoints=2,
            ncol=4,
            loc=2,
            bbox_to_anchor=(0 - ll, 1.03 + bb, 0, 0),
            fontsize=10,
            handlelength=2,
        )
        if plot_history:
            cax = fig.add_axes([ll + ww + 0.02, bb, 0.03, hh])
            cbar = ColorbarBase(ax=cax, cmap=cmap, alpha=0.6, ticks=[0, 0.5, 1])
            cbar.set_ticks(ticks=[0, 0.5, 1], labels=["Low", "Normal", "High"])
            cbar.solids.set(alpha=0.5)

    if title:
        ax.set_title(title)

    # plt.text(-5,1.15, "©Eawag, 2023-11-08", color="gray")

    return ax
