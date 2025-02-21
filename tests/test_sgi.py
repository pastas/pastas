import pandas as pd
from matplotlib import pyplot as plt

import pastas as ps

fileurl = "https://raw.githubusercontent.com/pastas/pastas/master/doc/examples/data"
dataset = pd.read_csv(f"{fileurl}/head_nb1.csv", index_col=0, parse_dates=True)

series = pd.Series(dataset["head"] * 100.0, dtype="int64")


plt.figure(figsize=(20, 12))
# original series
plt.subplot(2, 1, 1)
plt.title("Original groundwater head time series")
plt.xlim(dataset.index.values[0], dataset.index.values[-1])
plt.plot(dataset.index.values, dataset["head"].values, "k")
plt.ylabel("groundwater head")
plt.grid(True)
# SGIs
plt.subplot(2, 1, 2)
plt.title("Standardized Groundwater Index")
plt.ylim(-3, 3)
for period in [0, 1, 2, 3, 4, "a"]:
    aLabel = "SGI-" + str(period)
    sgiArr = period
    try:
        # with dataset
        aLab = aLabel + "df"
        sgiArr = ps.stats.sgi(dataset, period=period)
        plt.plot(sgiArr.index.values, sgiArr.values, label=aLab)
        # with Series
        aLab = aLabel + "series"
        sgiArr = ps.stats.sgi(series, period=period)
        plt.plot(sgiArr.index.values, sgiArr.values, ls=":", label=aLab)
    except:
        # ruff check   gives error for line above
        #              E722 Do not use bare `except`
        if period in [1, 2, 3]:
            raise Exception("sgi should work for period=" + str(period))
    if period == 1:
        assert abs(sgiArr.iloc[1] - -0.295179) < 0.001

plt.xlim(dataset.index.values[0], dataset.index.values[-1])
plt.ylabel("SGI")
plt.grid(True)
plt.legend()
plt.show()
