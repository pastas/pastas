"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

import pastas as ps

fname = 'data/MenyanthesTest.men'
meny = ps.read.menydata(fname)

# Create the time series model\
H=meny.H['Obsevation well']
ml = ps.Model(H['values'])

# Add precipitation
IN = meny.IN['Precipitation']
# round to days (precipitation is measured at 9:00)
IN['values'].index = IN['values'].index.normalize()
IN2 = meny.IN['Evaporation']
# round to days (evaporation is measured at 1:00)
IN2['values'].index = IN2['values'].index.normalize()
ts = ps.Tseries2(IN['values'], IN2['values'], ps.Gamma, 'Recharge')
ml.add_tseries(ts)

# Add well extraction 1
IN = meny.IN['Extraction 1']
ts = ps.Tseries(IN['values'], ps.Hantush, 'Extraction_1', up=False, freq='MS', fillnan='bfill')
ml.add_tseries(ts)

# Add well extraction 2
IN = meny.IN['Extraction 2']
ts = ps.Tseries(IN['values'], ps.Hantush, 'Extraction_2', up=False, freq='MS', fillnan='bfill')
ml.add_tseries(ts)

# Add well extraction 3
IN = meny.IN['Extraction 3']
ts = ps.Tseries(IN['values'], ps.Hantush, 'Extraction_3', up=False, freq='MS', fillnan='bfill')
ml.add_tseries(ts)

# replace extraction 3 by a step-function, to test the step-tseries
# ts = TseriesStep(pd.Timestamp(1970,1,1), 'step', rfunc=Gamma, up=False)
# ml.add_tseries(ts)

# Add noise model
n = ps.NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plots.decomposition()