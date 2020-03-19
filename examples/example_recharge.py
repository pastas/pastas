"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pandas as pd

import pastas as ps

ps.set_log_level("ERROR")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True,
                  squeeze=True)

# Create the time series model
ml = ps.Model(obs, name="head")

# read weather data and make mm/d !!!
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True) * 1e3
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True) * 1e3

# Initialize recharge model and create stressmodel
rch = ps.rch.FlexModel()
# rch = ps.rch.Berendrecht()
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma, recharge=rch)

ml.add_stressmodel(sm)

# Solve the model, first we solve without a noise model to get better initial
# estimates of the parameters, and then with a noise model.
ml.solve(noise=False, report=False)
ml.solve(noise=True, initial=False)

ml.plots.results()
