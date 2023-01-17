"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

noise = False
fit_constant = False

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True).squeeze("columns")

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True).squeeze(
    "columns"
)

# Solve with a Exponential response function
ml1 = ps.Model(obs, name="Exp")
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential())
ml1.add_stressmodel(sm)
ml1.solve(noise=noise, fit_constant=fit_constant)

# Solve with a Gamma response function
ml2 = ps.Model(obs, name="Gamma")
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma())
ml2.add_stressmodel(sm)
ml2.solve(noise=noise, fit_constant=fit_constant)

# Solve with a Spline response function
ml3 = ps.Model(obs, name="Spline")
rfunc = ps.Spline(t=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=rfunc)
ml3.add_stressmodel(sm)
ml3.solve(noise=noise, fit_constant=fit_constant)

# Compare both models
axes = ps.plots.compare([ml1, ml2, ml3])
