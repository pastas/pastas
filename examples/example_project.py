"""This file contains an example of the use of the Project class.

R.A. Collenteur - Artesia Water 2017

"""

import pastas as ps

# Create a simple model taken from example.py
obs = ps.read_dino('data/B58C0698001_1.csv')
rain = ps.read_knmi('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variables='RD')
evap = ps.read_knmi('data/etmgeg_380.txt', variables='EV24')

# Create a Pastas Project
mls = ps.Project(name="test_project")

mls.add_series(obs, "GWL", kind="oseries", metadata=dict())
mls.add_series(rain, name="Prec", kind="prec", metadata=dict())
mls.add_series(evap, name="Evap", kind="evap", metadata=dict())

ml = mls.add_model(oseries="GWL")
sm = ps.StressModel2([mls.stresses.loc["Prec", "series"],
                      mls.stresses.loc["Evap", "series"]],
                     ps.Exponential, name='recharge')
ml.add_stressmodel(sm)
n = ps.NoiseModel()
ml.add_noisemodel(n)
ml.solve(freq="D", warmup=1000, report=False)

mls.dump("test_project.pas")
