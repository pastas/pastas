import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pastas.timeseries_utils import time_weighted_resample

# %% make a daily series from monthly (mostly) values
# without specifying the frequency of the original series
index = pd.date_range("2000-1-1", "2001-1-1", freq="MS")
series0 = pd.Series(np.random.rand(len(index)), index)
new_index = series0.resample("D").mean().index
series1 = time_weighted_resample(series0, new_index)

plt.figure()
series0.plot(label="Monthly")
series1.plot(label="Daily")
plt.legend()

# %% make a precipitation-series at 0:00 from values at 9:00
index = pd.date_range("2000-1-1 9:00", "2000-1-10 9:00")
series0 = pd.Series(np.random.rand(len(index)), index)
new_index = pd.date_range("2000-1-1 0:00", "2000-1-12 0:00")
series1 = time_weighted_resample(series0, new_index)

plt.figure()
series0.plot(label="Original (9:00)")
series1.plot(label="Resampled (0:00)")
plt.legend()
