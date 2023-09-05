"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd

import pastas as ps

ps.set_log_level("INFO")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")

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
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
ml.add_stressmodel(sm)

# Solve with a warmup of 1 day
ml.solve(warmup=1)

# Copy and solve with a warmup of 1 day and normalize_stresses
ml2 = ml.copy()
ml2.solve(warmup=1, normalize_stresses=True)

# compare both models
# ps.plots.compare([ml, ml2], split=True)
ml.plots.decomposition()
ml2.plots.decomposition()
