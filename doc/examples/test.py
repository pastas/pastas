# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatter
from pandas import Timestamp

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
    data["precipitation"]["Davos"],
    data["evaporation"]["Davos"],
    ps.Gamma(),
    temp=data["temperature"]["Davos"],
    recharge=ps.rch.FlexModel(snow=True),
    name="recharge",
)
ml.add_stressmodel(rm)
ml.solve()
ml.plots.results()


# %%
def _plot_response_in_results(
    ml, sm_name, block_or_step, rmin, rmax, axb, istress=None
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

    rmax = max(rmax, response.index.max())
    response.plot(ax=axb)
    if block_or_step == "block":
        title = "Block response"
        rmin = response.index[1]
        axb.set_xscale("log")
        axb.xaxis.set_major_formatter(LogFormatter())
    else:
        title = "Step response"
    axb.set_title(title, fontsize=plt.rcParams['legend.fontsize'])
    return axb, rmin, rmax


# %%
tmin = None
tmax = None
figsize: tuple = (10, 8)
split: bool = False
adjust_height: bool = True
return_warmup: bool = False
block_or_step: str = "step"
stderr: bool = True
fig = None
kwargs = {}

tmin = Timestamp(tmin) if tmin is not None else None
tmax = Timestamp(tmax) if tmax is not None else None

# Number of rows to make the figure with
o = ml.observations(tmin=tmin, tmax=tmax)
o_nu = ml.oseries.series.drop(o.index)
if return_warmup:
    o_nu = o_nu[tmin - ml.settings["warmup"] : tmax]
else:
    o_nu = o_nu[tmin:tmax]
sim = ml.simulate(tmin=tmin, tmax=tmax, return_warmup=return_warmup)
res = ml.residuals(tmin=tmin, tmax=tmax)
contribs = {
    x.name: x
    for x in ml.get_contributions(
        split=split, tmin=tmin, tmax=tmax, return_warmup=return_warmup
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

if adjust_height:
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
else:
    height_ratios = [2] + [1] * (len(contribs) + 1)

mosaic = [[x] for x in ylims]
for mos in mosaic:
    if "cont_" in mos[0]:
        mos.append(f"rfunc_{mos[0].split('_')[1]}")
    elif mos[0] in "sim" or "res":
        mos.append("tab")
mosaic = np.array(mosaic, dtype=str)

fig, axd = plt.subplot_mosaic(
    mosaic,
    width_ratios=[2, 1],
    height_ratios=height_ratios,
    figsize=figsize,
    constrained_layout=True,
    **kwargs,
)

# Main frame
axd["sim"].plot(o.index, o.values, linestyle="", marker=".", color="k")
# plot parts of the oseries that are not used in grey
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

# add rsq to simulation
r2 = ml.stats.rsq(tmin=tmin, tmax=tmax)
axd["sim"].plot(sim.index, sim.values, label=f"{sim.name} ($R^2$={r2:.2%})")
axd["sim"].legend(loc=(0, 1), ncol=2, frameon=False, numpoints=3)
axd["sim"].set_ylim(bottom=ylims["sim"][0], top=ylims["sim"][1])

# Residuals and noise
axd["res"].plot(res.index, res.values, color="k", label="Residuals")
if ml.settings["noise"] and ml.noisemodel:
    noise = ml.noise(tmin=tmin, tmax=tmax)
    axd["res"].plot(noise.index, noise.values, label="Noise")
axd["res"].axhline(0.0, color="k", linestyle="--", zorder=0)
axd["res"].legend(loc=(0, 1), ncol=2, frameon=False)

rmin, rmax = 0.0, 0.0
for sm_name, sm in ml.stressmodels.items():
    # plot the contribution
    axd[f"cont_{sm_name}"].plot(
        contribs[sm_name].index,
        contribs[sm_name].values,
        label=sm_name,
    )
    axb, rmin, rmax = _plot_response_in_results(
        ml=ml,
        sm_name=sm_name,
        block_or_step=block_or_step,
        rmin=rmin,
        rmax=rmax,
        axb=axd[f"rfunc_{contrib.name}"],
    )
    title = [stress.name for stress in sm.stress]
    if len(title) > 3:
        title = title[:3] + ["..."]
    axd[f"cont_{sm_name}"].set_title(f"Stresses: {title}", loc="right", fontsize=plt.rcParams['legend.fontsize'])

    axd[f"cont_{sm_name}"].legend(loc=(0, 1), ncol=1, frameon=False)
    axd[f"cont_{sm_name}"].set_ylim(ylims[f"cont_{contrib.name}"])

if axb is not None:
    axb.set_xlim(rmin, rmax)

if return_warmup:
    axd["sim"].set_xlim(tmin - ml.settings["warmup"], tmax)
else:
    axd["sim"].set_xlim(tmin, tmax)

for k in [k for k in axd if "rfunc_" in k]:
    axd[k].yaxis.tick_right()
for k in axd:
    axd[k].grid(True)

share_xaxes([axd[k] for k in mosaic[:, 0]])
axd["sim"].set_xlim(tmin, tmax)

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
    colWidths=[0.4, 0.3, 0.3] if stderr else [0.8, 0.3],
    colLabels=p.columns,
)
