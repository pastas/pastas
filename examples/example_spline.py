"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True,
                  squeeze=True)

# Create the time series model

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)

# Solve with a Gamma response function
ml = ps.Model(obs, name="Gamma")
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma,
                      name='recharge')
ml.add_stressmodel(sm)
ml.solve(noise=False)

# Solve with a Spline response function
ml2 = ps.Model(obs, name="Spline")
sm2 = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Spline,
                       name='recharge')
ml2.add_stressmodel(sm2)
ml2.solve(noise=False)

# Compare both models
ps.plots.compare([ml, ml2])

axes = ps.plots.compare([ml, ml2], block_or_step='block')
