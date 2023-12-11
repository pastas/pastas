import numpy as np
from pandas import Series, date_range

import pastas as ps


def test_summary():
    """Test all signatures for minimal functioning."""
    idx = date_range("2000", "2010")
    head = Series(index=idx, data=np.random.rand(len(idx)), dtype=float)
    ps.stats.signatures.summary(head)
