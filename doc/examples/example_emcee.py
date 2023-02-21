"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import corner
import emcee
import matplotlib.pyplot as plt
import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

head = pd.read_csv(
    "data/B32C0639001.csv", parse_dates=["date"], index_col="date"
).squeeze()

# Make this millimeters per day
evap = pd.read_csv("data/evap_260.csv", index_col=0, parse_dates=[0]).squeeze()
rain = pd.read_csv("data/rain_260.csv", index_col=0, parse_dates=[0]).squeeze()

ml = ps.Model(head)

# Select a recharge model
rch = ps.rch.FlexModel()

rm = ps.RechargeModel(rain, evap, recharge=rch, rfunc=ps.Gamma(), name="rch")
ml.add_stressmodel(rm)

ml.solve(noise=True, tmin="1990")

# Now run with spotpy
s = ps.EmceeSolve(moves=emcee.moves.DEMove(),
                  obj_func=ps.objfunc.GaussianLikelihoodAr1(),
                  progress_bar=True, parallel=False)
ml.solve(solver=s, initial=False, noise=False, tmin="1990", steps=500, tune=True)
ml.plot()

fig = plt.figure(figsize=(10,10))

labels = list(ml.parameters.index[ml.parameters.vary])
labels = labels + list(ml.fit.parameters.index.values)

axes = corner.corner(
    ml.fit.sampler.get_chain(flat=True),
    quantiles=[0.025, 0.975],
    labelpad=0.1,
    show_titles=True,
    label_kwargs=dict(fontsize=10),
    max_n_ticks=3,
    fig=fig,
    labels=labels,
)
