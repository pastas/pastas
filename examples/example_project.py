"""This file contains an example of the use of the Project class.

R.A. Collenteur - Artesia Water 2017

"""

import pastas as ps

# Create a simple model taken from example.py
obs = ps.read.dinodata('data/B58C0698001_1.csv')
rain = ps.read.knmidata('data/neerslaggeg_HEIBLOEM-L_967-2.txt', variable='RD')
evap = ps.read.knmidata('data/etmgeg_380.txt', variable='EV24')

# Create a Pastas Project
mls = ps.Project(name="test_project")

mls.add_series(obs.series, "GWL", kind="oseries", metadata=dict())
mls.add_series(rain.series, name="Prec", kind="prec", metadata=dict())
mls.add_series(evap.series, name="Evap", kind="evap", metadata=dict())

ml = mls.add_model(oseries="GWL")
ts = ps.StressModel2([mls.stresses.loc["Prec", "series"],
                      mls.stresses.loc["Evap", "series"]],
                     ps.Exponential, name='recharge')
ml.add_stressmodel(ts)
n = ps.NoiseModel()
ml.add_noisemodel(n)
ml.solve(freq="D", warmup=1000, report=False)

mls.dump("test_project.pas")
