"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.
"""

from pasta import *
from pasta.read.menydata import MenyData
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

fname = 'data/MenyanthesTest.men'
meny = MenyData(fname)

# Round the time of observations to days (using ceil, so the observation is
# never before the stress)
#meny.H[0].series.index = meny.H[0].series.index.ceil('d')

# Create the time series model
ml = Model(meny.H[0].series)

# Add drainage level
d = Constant(value=meny.H[0].series.mean())
ml.add_tseries(d)

# Add precipitation
IN = next(x for x in meny.IN if x.name == 'Precipitation')
# round to days (precipitation is measured at 9:00)
IN.series = IN.series.resample('d').bfill()
if False:
    ts = Tseries(IN.series, Gamma, IN.name)
else:
    IN2 = next(x for x in meny.IN if x.name == 'Evaporation')
    IN2.series = IN2.series.resample('d').bfill()
    ts = Tseries2(IN.series, IN2.series, Gamma, 'Recharge')
ml.add_tseries(ts)

# Add well extraction at Ossendrecht
IN = next(x for x in meny.IN if x.name == 'Extraction 1')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
ts = Tseries(-IN.series/1000, Gamma, IN.name)
ml.add_tseries(ts)

# Add well extraction at Huijbergen
IN = next(x for x in meny.IN if x.name == 'Extraction 2')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
ts = Tseries(-IN.series/1000, Gamma, IN.name)
ml.add_tseries(ts)

# Add well extraction at Essen
IN = next(x for x in meny.IN if x.name == 'Extraction 3')
# extraction amount counts for the previous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace(' ','_')
ts = Tseries(-IN.series/1000, Gamma, IN.name)
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot_decomposition()