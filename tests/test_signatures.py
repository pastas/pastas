import numpy as np
from pandas import Series, date_range

import pastas as ps


def test_summary() -> None:
    """Test all signatures for minimal functioning."""
    idx = date_range("2000", "2010")
    head = Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    ps.stats.signatures.summary(head)


def test_date_min() -> None:
    """Test the date_min signature to have a mean of 1 if the head is at a minimum at
    the first of january of every year in a time series.
    """
    idx = date_range("2000", "2010")
    head = Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    head[head.index.dayofyear == 1] = head.min()
    assert round(ps.stats.signatures.date_min(head)) == 1


def test_date_max() -> None:
    """Test the date_max signature to have a mean of 1 if the head is at a maximum at
    the first of january of every year in a time series.
    """
    idx = date_range("2000", "2010")
    head = Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    head[head.index.dayofyear == 1] = head.max()
    assert round(ps.stats.signatures.date_max(head)) == 1
