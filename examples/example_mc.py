"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd
import matplotlib.pyplot as plt

import pastas as ps
from pastas.stats import nse, kge_2012

ps.set_log_level("ERROR")

# read observations and create the time series model and make meters
obs = pd.read_csv("data/B32C0639001.csv", parse_dates=['date'],
                  index_col='date', squeeze=True)

# Create the time series model
ml = ps.Model(obs, name="head")

# read weather data and make mm/d !
evap = ps.read_knmi("data/etmgeg_260.txt", variables="EV24").series * 1e3
rain = ps.read_knmi("data/etmgeg_260.txt", variables="RH").series * 1e3

# Initialize recharge model and create stressmodel
rch = ps.rch.FlexModel()
# rch = ps.rch.Berendrecht()
# rch = ps.rch.Linear()
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential, recharge=rch)

ml.add_stressmodel(sm)

ml.solve(noise=True, tmin="1990")
def target(p):
    obs = ml.observations()
    sim = ml.simulate(p=p)
    return nse(obs=obs, sim=sim)

# Solve
for param in ml.parameters.index:
    ml.set_parameter(param, vary=False)

ml.set_parameter("recharge_ks", vary=True)
#ml.set_parameter("recharge_srmax", vary=True)

#ml.set_parameter("constant_d", pmin=0, pmax=1)
#ml.set_parameter("recharge_A", pmin=0, pmax=1)
#ml.set_parameter("recharge_n", pmin=0, pmax=10)

#ml.set_parameter("recharge_gamma", pmin=0, pmax=10)
#ml.set_parameter("recharge_ks", vary=False)

ml.solve(solver=ps.MonteCarlo, target=target, n=int(5e4), noise=False,
         initial=False)
ml.plot()

fig, axes = plt.subplots(3, 3, sharey=True)
axes = axes.flatten()
for i, (name, par) in enumerate(ml.fit.parameters.iteritems()):
    axes[i].plot(par, ml.fit.obj, marker=".", linestyle=' ')
    axes[i].set_xlabel(name)
    axes[i].axvline(ml.parameters.loc[name, "optimal"], color="k",
                    linestyle="--")
for i in range(0, 9, 3):
    axes[i].set_ylabel("NSE [-]")
axes[-1].set_ylim(0, 1)
plt.tight_layout()

# Contour plot of the parameter landscape
x = ml.fit.parameters.recharge_ks[ml.fit.obj>-0]
y = ml.fit.parameters.recharge_srmax[ml.fit.obj>-0]
z = ml.fit.obj[ml.fit.obj > -0]

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
plt.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
cntr = ax.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
cb = fig.colorbar(cntr, ax=ax, label="NSE [-]")
ax.set_xlabel(x.name)
ax.set_ylabel(y.name)

