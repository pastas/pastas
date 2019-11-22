"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pandas as pd

import pastas as ps

# read observations
head = pd.read_csv("notebooks/data_notebook_7/head_wellex.csv",
                   index_col="Date", parse_dates=True)

# Create the time series model
ml = ps.Model(head, name="head")

# read weather data
rain = pd.read_csv("notebooks/data_notebook_7/prec_wellex.csv",
                   index_col="Date", parse_dates=True)
evap = pd.read_csv("notebooks/data_notebook_7/evap_wellex.csv",
                   index_col="Date", parse_dates=True)

# Create stress
rm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                      recharge=ps.rch.FlexModel(preferential=False),
                      name='recharge')
ml.add_stressmodel(rm)

well = pd.read_csv("notebooks/data_notebook_7/well_wellex.csv",
                   index_col="Date", parse_dates=True)/1e6
sm = ps.StressModel(well, rfunc=ps.Exponential, name="well", up=False)
ml.add_stressmodel(sm)

# Solve
ml.solve(noise=True)
ml.plots.results()

df = ml.fit.prediction_interval()
