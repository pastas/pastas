import numpy as np
from pandas import Series, date_range

import pastas as ps


def test_summary():
    """Test all signatures for minimal functioning."""
    idx = date_range("2000", "2010")
    head = Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    ps.stats.signatures.summary(head)


def test_colwell_components(collwell_data: Series) -> None:
    ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)
    return


def test_colwell_predictability(collwell_data: Series) -> None:
    """Example Tree C from the publication."""
    p = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[0]
    assert round(p, 2) == 1.0


def test_colwell_constancy(collwell_data: Series) -> None:
    """Example Tree C from the publication."""
    c = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[1]
    assert round(c, 2) == 0.42


def test_colwell_contingency(collwell_data: Series) -> None:
    """Example Tree C from the publication."""
    m = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[2]
    assert round(m, 2) == 0.58
