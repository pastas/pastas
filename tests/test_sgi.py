import pandas as pd

import pastas as ps

fileurl = "https://raw.githubusercontent.com/pastas/pastas/master/doc/examples/data"
dataset = pd.read_csv(f"{fileurl}/head_nb1.csv", index_col=0, parse_dates=True)

series = pd.Series(dataset["head"] * 100.0, dtype="int64")

for period in [0, 1, 2, 3, 4, "a"]:
    aLabel = "SGI-" + str(period)
    sgiArr = period
    try:
        # with dataset
        sgiArr = ps.stats.sgi(dataset, period=period)
        # with Series
        sgiArr = ps.stats.sgi(series, period=period)
    except:
        # ruff check   gives error for line above
        #              E722 Do not use bare `except`
        if period in [1, 2, 3]:
            raise Exception("sgi should work for period=" + str(period))
    if period == 1:
        assert abs(sgiArr.iloc[1] - -0.295179) < 0.001
