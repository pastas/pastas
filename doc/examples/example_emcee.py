"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import corner
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# Set the initial parameters to a normal distribution
# for name in ml.parameters.index:
#     ml.set_parameter(name, dist="norm")

# Create the EmceeSolver with some settings
s = ps.EmceeSolve(
    moves=emcee.moves.DEMove(),
    objective_function=ps.objfunc.GaussianLikelihoodAr1(),
    progress_bar=True,
    parallel=True,
)

# Use the solver to run MCMC
ml.solve(
    solver=s,
    initial=False,
    fit_constant=False,
    noise=False,
    tmin="1990",
    steps=15000,
    tune=True,
)

# Plot results and uncertainty
ax = ml.plot()

chain = ml.fit.sampler.get_chain(flat=True, discard=3000)
inds = np.random.randint(len(chain), size=100)
for ind in inds:
    params = chain[ind]
    p = ml.parameters.optimal.copy().values
    p[ml.parameters.vary == True] = params[: -ml.fit.objective_function.nparam]
    ml.simulate(p, tmin="1990").plot(c="gray", alpha=0.1, zorder=-1)

# Corner plot of the results
fig = plt.figure(figsize=(10, 10))

labels = list(ml.parameters.index[ml.parameters.vary])
labels = labels + list(ml.fit.parameters.index.values)

axes = corner.corner(
    ml.fit.sampler.get_chain(flat=True, discard=4000),
    quantiles=[0.025, 0.975],
    labelpad=0.1,
    show_titles=True,
    label_kwargs=dict(fontsize=10),
    max_n_ticks=3,
    fig=fig,
    labels=labels,
)
