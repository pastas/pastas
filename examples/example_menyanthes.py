"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

from pastas import *

fname = 'data/MenyanthesTest.men'
meny = read.menydata(fname)

# Create the time series model\
H=meny.H['Obsevation well']
ml = Model(H['values'])

freq='W'

# Add precipitation
IN = meny.IN['Precipitation']
# round to days (precipitation is measured at 9:00)
IN['values'] = IN['values'].resample(freq).bfill().dropna()
IN2 = meny.IN['Evaporation']
# round to days (evaporation is measured at 1:00)
IN2['values'] = IN2['values'].resample(freq).bfill().dropna()
ts = Tseries2(IN['values'], IN2['values'], Gamma, 'Recharge')
ml.add_tseries(ts)

# Add well extraction 1
IN = meny.IN['Extraction 1']
# extraction amount counts for the previous month
IN['values'] = IN['values'].resample(freq).bfill().dropna()
ts = Tseries(IN['values'], Gamma, 'Extraction_1', up=False)
ml.add_tseries(ts)
#
# Add well extraction 2
IN = meny.IN['Extraction 2']
# extraction amount counts for the previous month
IN['values'] = IN['values'].resample(freq).bfill().dropna()
ts = Tseries(IN['values'], Gamma, 'Extraction_2', up=False)
ml.add_tseries(ts)
#
# Add well extraction 3
IN = meny.IN['Extraction 3']
# extraction amount counts for the previous month
IN['values'] = IN['values'].resample(freq).bfill().dropna()
ts = Tseries(IN['values'], Gamma, 'Extraction_3', up=False)
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot_decomposition()