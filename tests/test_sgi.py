import pytest
from pandas import Series, concat

import pastas as ps

tol = 1e-3


def test_sgi_period1(head: Series) -> None:
    sgi = ps.stats.sgi(head, timescale_months=1)
    assert pytest.approx(sgi.iat[30], tol) == -0.812


def test_sgi_df(head: Series) -> None:
    df = concat([head, head], axis=1)
    sgi = ps.stats.sgi(df, timescale_months=1)
    assert pytest.approx(sgi.iat[30, 0], tol) == -0.812
    assert pytest.approx(sgi.iat[30, 1], tol) == -0.812


def test_sgi_period2(head: Series) -> None:
    sgi = ps.stats.sgi(head, timescale_months=2)
    assert pytest.approx(sgi.iat[30], tol) == -1.258


def test_sgi_perioderror(head: Series) -> None:
    with pytest.raises(ValueError):
        _ = ps.stats.sgi(head, timescale_months=4)
