"""
This test file is meant for developing purposes, providing an easy method to
test the functioning of Pastas recharge module during development.

Author: R.A. Collenteur, University of Graz.

"""
import pandas as pd

import pastas as ps

# read observations
head = pd.read_csv(
    "data_notebook_5/head_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")

# Create the time series model
ml = ps.Model(head, name="head")

# read weather data
rain = pd.read_csv(
    "data_notebook_5/prec_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")
evap = pd.read_csv(
    "data_notebook_5/evap_wellex.csv", index_col="Date", parse_dates=True
).squeeze("columns")

# Create stress
rm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
ml.add_stressmodel(rm)

well = (
    pd.read_csv("data_notebook_5/well_wellex.csv", index_col="Date", parse_dates=True)
    / 1e6
).squeeze("columns")
sm = ps.StressModel(well, rfunc=ps.Exponential(), name="well", up=False)
ml.add_stressmodel(sm)

# Solve
ml.solve(noise=True)
ml.plots.results()
