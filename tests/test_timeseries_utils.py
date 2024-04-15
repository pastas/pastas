import numpy as np
import pandas as pd
import pytest

import pastas as ps


def test_frequency_is_supported():
    ps.ts._frequency_is_supported("D")
    ps.ts._frequency_is_supported("7D")
    with pytest.raises(Exception):
        ps.ts._frequency_is_supported("SMS")


def test_get_stress_dt():
    assert ps.ts._get_stress_dt("D") == 1.0
    assert ps.ts._get_stress_dt("7D") == 7.0
    assert ps.ts._get_stress_dt("W") == 7.0
    assert ps.ts._get_stress_dt("SMS") == 15.0


def test_time_series_sampling_methods():
    # some helper functions to compute differences in performance
    def values_kept(s, original):
        diff = set(original.dropna().values) & set(s.dropna().values)
        return len(diff)

    def n_duplicates(s):
        return (s.value_counts() >= 2).sum()

    # Create timeseries
    freq = "2h"
    freq2 = "1h"
    idx0 = pd.date_range("2000-01-01 18:00:00", freq=freq, periods=3).tolist()
    idx1 = pd.date_range("2000-01-02 01:30:00", freq=freq2, periods=10).tolist()
    idx0 = idx0 + idx1
    idx0[3] = pd.Timestamp("2000-01-02 01:31:00")
    series = pd.Series(index=idx0, data=np.arange(len(idx0), dtype=float))
    series.iloc[8:10] = np.nan

    # Create equidistant timeseries
    s_pd1 = ps.ts.pandas_equidistant_sample(series, freq)
    s_pd2 = ps.ts.pandas_equidistant_nearest(series, freq)
    s_pd3 = ps.ts.pandas_equidistant_asfreq(series, freq)
    s_pastas1 = ps.ts.get_equidistant_series_nearest(
        series, freq, minimize_data_loss=False
    )
    s_pastas2 = ps.ts.get_equidistant_series_nearest(
        series, freq, minimize_data_loss=True
    )

    dfall = pd.concat(
        [series, s_pd1, s_pd2, s_pd3, s_pastas1, s_pastas2], axis=1, sort=True
    )
    dfall.columns = [
        "original",
        "pandas_equidistant_sample",
        "pandas_equidistant_nearest",
        "pandas_equidistant_asfreq",
        "get_equidistant_series_nearest (default)",
        "get_equidistant_series_nearest (minimize data loss)",
    ]
    valueskept = dfall.apply(values_kept, args=(dfall["original"],))
    duplicates = dfall.apply(n_duplicates)
    print(valueskept)
    assert np.all(valueskept.values == np.array([11, 4, 7, 8, 8, 9]))
    assert np.all(duplicates.values == np.array([0, 0, 2, 0, 0, 0]))
