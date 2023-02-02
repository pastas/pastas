import numpy as np
import pandas as pd

import pastas as ps

from .fixtures import collwell_data


def test_summary():
    """Test all signatures for minimal functioning."""
    idx = pd.date_range("2000", "2010")
    head = pd.Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    ps.stats.signatures.summary(head)


def test_colwell_components(collwell_data) -> None:
    ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)
    return


def test_colwell_predictability(collwell_data) -> None:
    """Example Tree C from the publication."""
    p = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[0]
    assert p.round(2) == 1.0


def test_colwell_constancy(collwell_data) -> None:
    """Example Tree C from the publication."""
    c = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[1]
    assert c.round(2) == 0.42


def test_colwell_contingency(collwell_data) -> None:
    """Example Tree C from the publication."""
    m = ps.stats.signatures.colwell_components(collwell_data, freq="4M", bins=3)[2]
    assert m.round(2) == 0.58
