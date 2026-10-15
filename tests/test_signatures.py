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


def test_bimodality_coefficient() -> None:
    """Test the bimodality coefficient against the example distributions in Table A1
    of Pfister et al. (2013), https://doi.org/10.3389/fpsyg.2013.00700.
    """
    freqs = {
        0.34: [3, 5, 5, 10, 17, 20, 17, 10, 5, 5, 3],
        0.79: [2, 26, 14, 6, 2, 0, 2, 6, 14, 26, 2],
        0.73: [2, 3, 3, 3, 3, 4, 5, 11, 21, 41, 4],
        0.67: [2, 3, 6, 17, 3, 4, 5, 12, 14, 30, 4],
    }
    for expected, freq in freqs.items():
        data = np.repeat(np.arange(1, 12), freq).astype(float)
        head = Series(index=date_range("2000", periods=data.size), data=data)
        bc = ps.stats.signatures.bimodality_coefficient(head, normalize=False)
        assert round(bc, 2) == expected
