import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pastas.utils import timestep_weighted_resample, \
    timestep_weighted_resample_fast

# make a daily series from monthly (mostly) values, without specifying the
# frequency of the original series
# series0 = pd.read_csv("data/tswr1.csv", index_col=0, parse_dates=True,
#                      squeeze=True)
index = pd.date_range('2000-1-1', '2001-1-1', freq='MS')
series0 = pd.Series(np.random.rand(len(index)), index)
series = series0.resample('d').mean()
series = timestep_weighted_resample(series0, series.index)
series2 = timestep_weighted_resample_fast(series0, 'd')

plt.figure()
series0.plot(label='Monthly (mostly)')
series.plot(label='Daily')
series2.plot(label='Daily (fast)', linestyle='--')
plt.legend()

# make a precipitation-series at 0:00 from values at 9:00
# series0 = pd.read_csv("data/tswr2.csv", index_col=0, parse_dates=True,
#                      squeeze=True)
index = pd.date_range('2000-1-1 9:00', '2000-1-10 9:00')
series0 = pd.Series(np.random.rand(len(index)), index)
series = timestep_weighted_resample(series0, series0.index.normalize())
series2 = timestep_weighted_resample_fast(series0, 'd')
plt.figure()
series0.plot(label='Original (9:00)')
series.plot(label='Resampled (0:00)')
series2.plot(label='Resampled (0:00, fast)', linestyle='--')
plt.legend()
