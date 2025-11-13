# %%
"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""

import numpy as np
import pandas as pd

import pastas as ps

ps.set_log_level("WARNING")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")
t = (obs.index - obs.index[0]).to_numpy() / np.timedelta64(1, "D")
a = 0.05 / 365  # 5 cm/year
trend = pd.Series(a * t, index=obs.index)
obs = obs + trend  # add trend to observations


# Create the time series model
ml = ps.Model(obs, name="head")

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)

# Create stress
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="recharge")
ml.add_stressmodel(sm)

# ml.set_parameter("recharge_a", pmax=100)

# Solve
ml.solve(report=False)
ml.plot()
