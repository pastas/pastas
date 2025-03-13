# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter
from pandas import Timestamp
from typing import Literal
import pastas as ps
from pastas.plotting.plotutil import (
    _get_height_ratios,
    _table_formatter_params,
    _table_formatter_stderr,
    share_xaxes,
)

# %%
data = ps.load_dataset("collenteur_2023")
# %%
ml = ps.Model(data["heads"]["Davos"].dropna())
rm = ps.RechargeModel(
    data["precipitation"]["Davos"].copy().rename("davos_rain"),
    data["evaporation"]["Davos"].copy().rename("davos_evap"),
    ps.Gamma(),
    temp=data["temperature"]["Davos"].copy().rename("davos_temp"),
    recharge=ps.rch.FlexModel(snow=True),
    name="rech",
)
sm = ps.StepModel("2010", "step", rfunc=ps.Exponential())
ml.add_stressmodel([rm, sm])
ml.add_noisemodel(ps.ArNoiseModel())
ml.solve()
ml.plots.results()


# %%
def _plot_response_in_results(
    ml: ps.Model, sm_name: str, block_or_step: Literal["step", "block"], ax: plt.Axes, istress: int | None = None
):
    """Internal method to plot the response of a Stressmodel in the results-plot"""
    rkwargs = {}
    if ml.stressmodels[sm_name].rfunc is not None:
        if isinstance(ml.stressmodels[sm_name].rfunc, ps.HantushWellModel):
            rkwargs = {"warn": False}
            if istress is None:
                # show the response of the first well, which gives more information than istress = None
                istress = 0
    response = ml._get_response(
        block_or_step=block_or_step,
        name=sm_name,
        add_0=True,
        istress=istress,
        **rkwargs,
    )
    label = f"{block_or_step.capitalize()} response"
    ax.plot(response.index, response.values, label=label)
    if block_or_step == "block":
        ax.set_xlim(left=response.index[1])
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(LogFormatter())
    else:
        ax.set_xlim(left=response.index[0])

# %%
tmin = None
tmax = None
figsize = None
split: bool = False
return_warmup: bool = False
block_or_step: str = "step"
stderr: bool = True
layout: Literal["constrained", "tight", "compressed", "none"] = "constrained"
fig_kwargs = {}

tmin = Timestamp(tmin) if tmin is not None else None
tmax = Timestamp(tmax) if tmax is not None else None

# get simulated time series
o = ml.observations(tmin=tmin, tmax=tmax)
o_nu = ml.oseries.series.drop(o.index)
o_nu = o_nu[tmin - ml.settings["warmup"] : tmax] if return_warmup else o_nu[tmin:tmax]
sim = ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
res = ml.residuals(tmin=tmin, tmax=tmax)
contribs = {
    x.name: x
    for x in ml.get_contributions(
        tmin=tmin, tmax=tmax, return_warmup=return_warmup
    )
}

ylims = {
    "sim": [
        min([sim.min(), o[tmin:tmax].min(), o_nu.min()]),
        max([sim.max(), o[tmin:tmax].max(), o_nu.max()]),
    ],
    "res": [res.min(), res.max()],
}
for k, ylim in ylims.items():
    yl_diff = (ylim[1] - ylim[0]) * 0.025
    ylims[k] = [ylim[0] - yl_diff, ylim[1] + yl_diff]

for cname, contrib in contribs.items():
    hs = contrib.loc[tmin:tmax]
    if hs.empty:
        if contrib.empty:
            ylim_c = [0.0, 0.0]
        else:
            ylim_c = [contrib.min(), hs.max()]
    else:
        ylim_c = [hs.min(), hs.max()]
    ylims[f"cont_{cname}"] = ylim_c
height_ratios = _get_height_ratios(list(ylims.values()))

mosaic = [[x] for x in ylims]
for mos in mosaic:
    if "cont_" in mos[0]:
        mos.append(f"rfunc_{mos[0].split('_')[1]}")
    elif mos[0] in "sim" or "res":
        mos.append("tab")
mosaic = np.array(mosaic, dtype=str)

if "width_ratios" not in fig_kwargs:
    fig_kwargs["width_ratios"] = [2.5, 1.0]

fig, axd = plt.subplot_mosaic(
    mosaic,
    height_ratios=height_ratios,
    figsize=(10, 4 + 2 * len(contribs)),
    layout=layout,
    **fig_kwargs,
)

# plot observations and simulation
axd["sim"].plot(o.index, o.values, linestyle="", marker=".", color="k")
if not o_nu.empty:
    axd["sim"].plot(
        o_nu.index,
        o_nu.values,
        linestyle="",
        marker=".",
        color="grey",
        label="",
        zorder=-1,
    )
axd["sim"].plot(sim.index, sim.values, label=f"{sim.name} ($R^2$={ml.stats.rsq(tmin=tmin, tmax=tmax):.2%})")
axd["sim"].legend(loc=(0, 1), ncol=2, frameon=False, numpoints=3)
axd["sim"].set_ylim(bottom=ylims["sim"][0], top=ylims["sim"][1])

# plot residuals (and noise if present)
axd["res"].plot(res.index, res.values, color="k", label="Residuals")
if ml.settings["noise"] and ml.noisemodel:
    noise = ml.noise(tmin=tmin, tmax=tmax)
    axd["res"].plot(noise.index, noise.values, label="Noise")
axd["res"].axhline(0.0, color="k", linestyle="--", zorder=0)
axd["res"].legend(loc=(0, 1), ncol=2, frameon=False)

# plot the contributions and resposnes of the stressmodels
for sm_name, sm in ml.stressmodels.items():
    axd[f"cont_{sm_name}"].plot(
        contribs[sm_name].index,
        contribs[sm_name].values,
        label=sm_name,
    )
    title = [stress.name for stress in sm.stress]
    if len(title) > 3:
        title = title[:3] + ["..."]
    if title:
        axd[f"cont_{sm_name}"].set_title(f"Stresses: " + str(title).replace("'", ""), loc="right", fontsize=plt.rcParams['legend.fontsize'])
    axd[f"cont_{sm_name}"].legend(loc=(0, 1), ncol=1, frameon=False)
    axd[f"cont_{sm_name}"].set_ylim(ylims[f"cont_{sm_name}"])
    _plot_response_in_results(
        ml=ml,
        sm_name=sm_name,
        block_or_step=block_or_step,
        ax=axd[f"rfunc_{sm_name}"],
    )

# share x-axes of simulation, residuals and contributions
share_xaxes([axd[k] for k in mosaic[:, 0]])
axd["sim"].set_xlim(tmin - ml.settings["warmup"], tmax) if return_warmup else axd["sim"].set_xlim(tmin, tmax)

# share x-axes of the responses
response_axes = [axd[k] for k in axd if "rfunc_" in k]
response_xlims = [ax.get_xlim() for ax in response_axes]
response_xlim_left = min(x[0] for x in response_xlims)
response_xlim_right = max(x[1] for x in response_xlims)
share_xaxes(response_axes)
for i, ax in enumerate(response_axes):
    if i == 0:
        ax.legend(loc=(0,1), frameon=False)
    ax.yaxis.tick_right()
    ax.set_xlim(left=response_xlim_left, right=response_xlim_right)

# add grid
for k in axd:
    axd[k].grid(True)

# plot parameters table
axd["tab"].set_title(f"Model parameters ($n_c$={ml.parameters.vary.sum()})", loc="left", fontsize=plt.rcParams['legend.fontsize'])
p = ml.parameters.loc[:, ["name"]].copy()
p.loc[:, "name"] = p.index
p.loc[:, "optimal"] = ml.parameters.loc[:, "optimal"].apply(_table_formatter_params)
if stderr:
    stderrper = ml.parameters.loc[:, "stderr"] / ml.parameters.loc[:, "optimal"]
    p.loc[:, "stderr"] = stderrper.abs().apply(_table_formatter_stderr)

axd["tab"].axis("off")
axd["tab"].table(
    bbox=(0.0, 0.0, 1.0, 1.0),
    cellText=p.values,
    colWidths=[p[col].str.len().max() for col in p.columns],
    colLabels=p.columns,
)
