import numpy as np
import pandas as pd

import pastas as ps


def acf_func(**kwargs):
    index = pd.to_datetime(np.arange(0, 100, 1), unit="D", origin="2000")
    data = np.sin(np.linspace(0, 10 * np.pi, 100))
    r = pd.Series(data=data, index=index)
    acf_true = np.cos(np.linspace(0, np.pi, 11))
    acf = ps.stats.acf(r, lags=np.arange(0.0, 11), **kwargs).values
    return acf, acf_true


def test_acf_rectangle():
    acf, acf_true = acf_func(bin_method="rectangle")
    assert abs((acf - acf_true)).max() < 0.05


def test_acf_gaussian():
    acf, acf_true = acf_func(bin_method="rectangle")
    assert abs((acf - acf_true)).max() < 0.05


def test_runs_test():
    """
    http://www.itl.nist.gov/div898/handbook/eda/section3/eda35d.htm
    True Z-statistic = 2.69
    Read NIST test data
    """
    data = pd.read_csv("tests/data/nist.csv")
    _, test, _ = ps.stats.runs_test(data)
    assert test[0] - 2.69 < 0.02
