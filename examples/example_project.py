"""This file contains an example of the use of the Project class.

R.A. Collenteur - Artesia Water 2017

"""

import pandas as pd

import pastas as ps

# Create a simple model taken from example.py
obs = pd.read_csv("data/head_nb1.csv", index_col=0, parse_dates=True,
                  squeeze=True)
rain = pd.read_csv("data/rain_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)
evap = pd.read_csv("data/evap_nb1.csv", index_col=0, parse_dates=True,
                   squeeze=True)

# Create a Pastas Project
mls = ps.Project(name="test_project")

mls.add_oseries(obs, "GWL", metadata={})
mls.add_stress(rain, name="Prec", kind="prec")
mls.add_stress(evap, name="Evap", kind="evap")

ml = mls.add_model(oseries="GWL")

mls.add_recharge(ml)
mls.solve_models()

mls.to_file("test_project.pas")
