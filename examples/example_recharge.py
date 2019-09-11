"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pandas as pd

import pastas as ps
from pastas.recharge import FlexModel, Berendrecht

ps.set_log_level("ERROR")

# read observations and create the time series model
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True,
                  squeeze=True)

# Create the time series model
ml = ps.Model(obs, name="head")

# read weather data
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)

# Initialize recharge model and create stressmodel
rch = FlexModel(preferential=False)
rch = Berendrecht()
sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                      recharge=rch, name='recharge')
ml.add_stressmodel(sm)

# Solve
ml.solve(noise=False)
ml.plots.results()
