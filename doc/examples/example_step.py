"""
This is an example with a step in the observations, which we add artificially.
We model this step through a StepModel.

"""

import pandas as pd

import pastas as ps

# read observations
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")
# add 10 cm to the series from 2007
obs["2007":] = obs["2007":] + 1.5

# Create the time series model
ml = ps.Model(obs)
ps.ArNoiseModel(model=ml)

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)

# create stress
sm = ps.RechargeModel(
    model=ml,
    prec=rain,
    evap=evap,
    rfunc=ps.Exponential(),
    recharge=ps.rch.Linear(),
    name="recharge",
)

# add a stepmodel with an exponential response
sm = ps.stressmodels.StepModel(model=ml, tstart="2007", name="step", rfunc=ps.One())

# solve
ml.solve()
ml.plots.decomposition()
