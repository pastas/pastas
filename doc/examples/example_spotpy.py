"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.

"""
import pandas as pd
import spotpy

import pastas as ps

ps.set_log_level("ERROR")

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

# Solve to get stderr and good starting point
ml.solve()

# Now run with spotpy
s = ps.SpotpySolve(algorithm=spotpy.algorithms.dream,
                   obj_func=spotpy.likelihoods.generalizedLikelihoodFunction)
ml.solve(solver=s, initial=False, noise=False,
         repetitions=5000, runs_after_convergence=1000)
ml.plot()
