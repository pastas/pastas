import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pastas.timeseries_utils import (
    timestep_weighted_resample,
)

# make a daily series from monthly (mostly) values, without specifying the
# frequency of the original series
index = pd.date_range("2000-1-1", "2001-1-1", freq="MS")
series0 = pd.Series(np.random.rand(len(index)), index)
new_index = series0.resample("D").mean().index
series1 = timestep_weighted_resample(series0, new_index, fast=False)
series2 = timestep_weighted_resample(series0, new_index, fast=True)

plt.figure()
series0.plot(label="Monthly")
series1.plot(label="Daily")
series2.plot(label="Daily (fast)", linestyle="--")
plt.legend()

# make a precipitation-series at 0:00 from values at 9:00
index = pd.date_range("2000-1-1 9:00", "2000-1-10 9:00")
series0 = pd.Series(np.random.rand(len(index)), index)
series1 = timestep_weighted_resample(series0, series0.index.normalize(), fast=False)
series2 = timestep_weighted_resample(series0, series0.index.normalize(), fast=True)

plt.figure()
series0.plot(label="Original (9:00)")
series1.plot(label="Resampled (0:00)")
series2.plot(label="Resampled (0:00, fast)", linestyle="--")
plt.legend()
