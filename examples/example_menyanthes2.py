"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

import pastas as ps

fname = 'data/MenyanthesTest.men'
meny = ps.read.MenyData(fname)

# Create the time series model\
H=meny.H['Obsevation well']
ml = ps.Model(H['values'])

# round to days (precipitation is measured at 9:00)
IN = meny.IN['Precipitation']
IN['values'].index = IN['values'].index.normalize()

#round to days (evaporation is measured at 1:00)
IN2 = meny.IN['Evaporation']
IN2['values'].index = IN2['values'].index.normalize()

sm = ps.StressModel2([IN['values'], IN2['values']], ps.Gamma, 'Recharge')
ml.add_stressmodel(sm)

settings = dict(freq='W')

# Add well extraction 1
IN = meny.IN['Extraction 1']
sm = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_1', up=False,
                    kind="well", settings=settings)
ml.add_stressmodel(sm)

# Add well extraction 2
IN = meny.IN['Extraction 2']
sm = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_2', up=False,
                    kind="well", settings=settings)
ml.add_stressmodel(sm)

#Add well extraction 3
# IN = meny.IN['Extraction 3']
# print(IN['values'].index.min())
# ts = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_3', up=False,
#                 kind="well", settings=settings)
# ml.add_tseries(ts)

# replace extraction 3 by a step-function, to test the step-tseries
# ts = TseriesStep(pd.Timestamp(1970,1,1), 'step', rfunc=Gamma, up=False)
# ml.add_tseries(ts)

# Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plots.decomposition()
