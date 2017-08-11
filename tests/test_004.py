# Test the changing of the frequency

from pastas.timeseries import TimeSeries
import pandas as pd

i = pd.date_range("01-01-2000", periods=10, freq="W")
s = pd.Series(pd.np.random.rand(10), index=i)

x = TimeSeries(s, freq="W", type="oseries", sample_up="bfill")

