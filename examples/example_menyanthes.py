"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

import pastas as ps

fname = 'data/MenyanthesTest.men'
meny = ps.read.MenyData(fname)

# Create the time series model
H = meny.H['Obsevation well']
ml = ps.Model(H['values'])

freq = 'W'

# Add precipitation
IN = meny.IN['Precipitation']['values']
IN.index = IN.index.round("D")
IN2 = meny.IN['Evaporation']['values']
IN2.index = IN2.index.round("D")
sm = ps.StressModel2([IN, IN2], ps.Gamma, 'Recharge')
ml.add_stressmodel(sm)

# Add well extraction 1
IN = meny.IN['Extraction 1']
# extraction amount counts for the previous month
sm = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_1', up=False,
                    kind="well", settings=dict(freq="W"))
ml.add_stressmodel(sm)

# Add well extraction 2
IN = meny.IN['Extraction 2']
# extraction amount counts for the previous month
sm = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_2', up=False,
                    kind="well", settings=dict(freq="W"))
ml.add_stressmodel(sm)

# # Add well extraction 3
# IN = meny.IN['Extraction 3']
# # extraction amount counts for the previous month
# sm = ps.StressModel(IN['values'], ps.Hantush, 'Extraction_3', up=False,
#                 kind="well", settings=dict(freq="W"))
# ml.add_tseries(sm)

# Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve(freq="D")
ml.plots.decomposition()
