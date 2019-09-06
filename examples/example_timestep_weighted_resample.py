import matplotlib.pyplot as plt
import pandas as pd

from pastas.utils import timestep_weighted_resample

# make a daily series from monthly (mostly) values, without specifying the
# frequency of the original series
series0 = pd.read_csv("data/tswr1.csv", index_col=0, parse_dates=True,
                      squeeze=True)
series = series0.resample('d').mean()
series = timestep_weighted_resample(series0, series.index)

plt.figure()
series0.plot(label='Monthly (mostly)')
series.plot(label='Daily')
plt.legend()

# make a precipitation-series at 0:00 from values at 9:00
series0 = pd.read_csv("data/tswr2.csv", index_col=0, parse_dates=True,
                      squeeze=True)
series = timestep_weighted_resample(series0, series0.index.normalize())
plt.figure()
series0.plot(label='Original (9:00)')
series.plot(label='Resampled (0:00)')
plt.legend()
