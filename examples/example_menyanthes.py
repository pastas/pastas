"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of Pastas during development.
"""

from pastas import *

fname = 'data/MenyanthesTest.men'
meny = read.menydata(fname)

# Create the time series model
ml = Model(meny.H[0].series)

# Add precipitation
IN = next(x for x in meny.IN if x.name == 'Precipitation')
# round to days (precipitation is measured at 9:00)
IN.series = IN.series.resample('d').bfill()
IN2 = next(x for x in meny.IN if x.name == 'Evaporation')
# round to days (evaporation is measured at 1:00)
IN2.series = IN2.series.resample('d').bfill()
ts = Tseries2(IN.series, IN2.series, Gamma, 'Recharge')
ml.add_tseries(ts)

# Add well extraction 1
IN = next(x for x in meny.IN if x.name == 'Extraction 1')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
# divide by thousand, as default starting parameters for gamma are wrong
ts = Tseries(IN.series/1000, Gamma, IN.name, up=False)
ml.add_tseries(ts)
#
# Add well extraction 2
IN = next(x for x in meny.IN if x.name == 'Extraction 2')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
# divide by thousand, as default starting parameters for gamma are wrong
ts = Tseries(IN.series/1000, Gamma, IN.name, up=False)
ml.add_tseries(ts)
#
# Add well extraction 3
IN = next(x for x in meny.IN if x.name == 'Extraction 3')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
# divide by thousand, as default starting parameters for gamma are wrong
ts = Tseries(IN.series/1000, Gamma, IN.name, up=False)
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot_decomposition()