"""
This test file is meant for developing purposes. Providing an easy method to
test the functioning of PASTA during development.
"""

from pasta import *
from pasta.read.menydata import MenyData

fname = 'data/MenyanthesTest.men'
meny = MenyData(fname)

# Round the time of observations to days (using ceil, so the observation is never before the stress)
meny.H[0].series.index = meny.H[0].series.index.ceil('d')

# Create the time series model
ml = Model(meny.H[0].series)

# Add drainage level
d = Constant(value=meny.H[0].series.min())
ml.add_tseries(d)

# Add precipitation
IN = next(x for x in meny.IN if x.name == 'Neerslag (Hoogerheide)')
# round to days (precipitation is measured at 9:00)
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace('(','')
IN.name = IN.name.replace(')','')
IN.name = IN.name.replace(' ','_')
ts = Tseries(IN.series, Gamma, IN.name)
ml.add_tseries(ts)

# Add well extraction at Ossendrecht
IN = next(x for x in meny.IN if x.name == 'Onttrekking (Ossendrecht)')
# extraction amound counts for the previeous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace('(','')
IN.name = IN.name.replace(')','')
IN.name = IN.name.replace(' ','_')
ts = Tseries(IN.series, Gamma, IN.name)
ml.add_tseries(ts)

# Add well extraction at Huijbergen
IN = next(x for x in meny.IN if x.name == 'Onttrekking (Huijbergen)')
# extraction amound counts for the previeous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace('(','')
IN.name = IN.name.replace(')','')
IN.name = IN.name.replace(' ','_')
ts = Tseries(IN.series, Gamma, IN.name)
ml.add_tseries(ts)

# Add well extraction at Essen
IN = next(x for x in meny.IN if x.name == 'Onttrekking (Essen)')
# extraction amound counts for the previeous month
IN.series = IN.series.resample('d').bfill()
IN.name = IN.name.replace('(','')
IN.name = IN.name.replace(')','')
IN.name = IN.name.replace(' ','_')
ts = Tseries(IN.series, Gamma, IN.name)
ml.add_tseries(ts)

# Add noise model
n = NoiseModel()
ml.add_noisemodel(n)

# Solve
ml.solve()
ml.plot_decomposition()
