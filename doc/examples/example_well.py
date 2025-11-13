# %%
"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""

import numpy as np
import pandas as pd

import pastas as ps

# read observations

ps.set_log_level("WARNING")

head = pd.read_csv(
    "data_notebook_5/head_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")

t = (head.index - head.index[0]).to_numpy() / np.timedelta64(1, "D")
a = -0.15 / 365  # 15 cm/year
trend = pd.Series(a * t, index=head.index)
head = head + trend  # add trend to observations
head = head.loc["2008":"2012"]

# Create the time series model
ml = ps.Model(head, name="head")
ml.add_noisemodel(ps.ArNoiseModel())

# read weather data
rain = pd.read_csv(
    "data_notebook_5/prec_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")
evap = pd.read_csv(
    "data_notebook_5/evap_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")

rain = rain.loc["1998":"2012"]
evap = evap.loc["1998":"2012"]

# Create stress
rm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
ml.add_stressmodel(rm)

well = (
    pd.read_csv("data_notebook_5/well_wellex.csv", index_col="Date", parse_dates=True)
    / 1e6
).squeeze("columns")
sm = ps.StressModel(well, rfunc=ps.Gamma(), name="well", up=False)
ml.add_stressmodel(sm)

# Solve
ml.solve(report=False)
ml.plots.results()
