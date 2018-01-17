import os
import pastas as ps
import matplotlib.pyplot as plt
from pastas.utils import timestep_weighted_resample

if not os.getcwd().endswith('examples'):
    os.chdir('examples')

fname = 'data/MenyanthesTest.men'
meny = ps.read.MenyData(fname)

# make a daily series from monthly (mostly) values, without specifying the frequency of the original series
series0 = meny.IN['Extraction 1']['values']
series = series0.resample('d').mean()
series = timestep_weighted_resample(series0, series.index)

plt.figure()
series0.plot(label='Monthly (mostly)')
series.plot(label='Daily')
plt.legend()

# make a precipitation-series at 0:00 from values at 9:00
series0 = meny.IN['Precipitation']['values']
series = timestep_weighted_resample(series0, series0.index.normalize())
plt.figure()
series0.plot(label='Original (9:00)')
series.plot(label='Resampled (0:00)')
plt.legend()
