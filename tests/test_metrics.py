import pandas as pd
import pytest

import pastas as ps

index = pd.date_range("2010-01-01", "2010-01-04", freq="D")
obs = pd.Series([10.0, 15.21, 20.0, 25.2], index=index)
sim = pd.Series([10.1, 15.0, 20.3, 25.0], index=index)
tol = 1e-6


def test_mae() -> None:
    mae = ps.stats.metrics.mae(obs=obs, sim=sim)
    assert pytest.approx(mae, tol) == 0.2025


def test_rmse() -> None:
    rmse = ps.stats.metrics.rmse(obs=obs, sim=sim)
    assert pytest.approx(rmse, tol) == 0.2145343


def test_sse() -> None:
    sse = ps.stats.metrics.sse(obs=obs, sim=sim)
    assert pytest.approx(sse, tol) == 0.1841


def test_pearsonr() -> None:
    r = ps.stats.metrics.pearsonr(obs=obs, sim=sim)
    assert pytest.approx(r, tol) == 0.999299


def test_evp() -> None:
    evp = ps.stats.metrics.evp(obs=obs, sim=sim)
    assert pytest.approx(evp, tol) == 99.85505


def test_nse() -> None:
    nse = ps.stats.metrics.nse(obs=obs, sim=sim)
    assert pytest.approx(nse, tol) == 0.998550


def test_nnse() -> None:
    nnse = ps.stats.metrics.nnse(obs=obs, sim=sim)
    assert pytest.approx(nnse, tol) == 0.998552


def test_rsq() -> None:
    rsq = ps.stats.metrics.rsq(obs=obs, sim=sim)
    assert pytest.approx(rsq, tol) == 0.99855


def test_bic() -> None:
    bic = ps.stats.metrics.bic(obs=obs, sim=sim)
    assert pytest.approx(bic, tol) == -10.9279878


def test_aic() -> None:
    aic = ps.stats.metrics.aic(obs=obs, sim=sim)
    assert pytest.approx(aic, tol) == -10.314282


def test_aicc() -> None:
    aicc = ps.stats.metrics.aicc(obs=obs, sim=sim)
    assert pytest.approx(aicc, tol) == -8.314282


def test_kge() -> None:
    kge = ps.stats.metrics.kge(obs=obs, sim=sim)
    assert pytest.approx(kge, tol) == 0.9923303


def test_kge_modified() -> None:
    kgem = ps.stats.metrics.kge(obs=obs, sim=sim, modified=True)
    assert pytest.approx(kgem, tol) == 0.99247


def test_picp() -> None:
    bounds = pd.DataFrame(
        {"lower": [9.0, 14.0, 19.0, 24.0], "upper": [11.0, 16.0, 21.0, 26.0]},
        index=index,
    )
    picp_value = ps.stats.metrics.picp(obs=obs, bounds=bounds)
    assert pytest.approx(picp_value, tol) == 1.0


def test_picp_partial_coverage() -> None:
    bounds = pd.DataFrame(
        {"lower": [9.0, 14.0, 19.0, 24.0], "upper": [10.5, 15.0, 20.5, 25.0]},
        index=index,
    )
    picp_value = ps.stats.metrics.picp(obs=obs, bounds=bounds)
    assert pytest.approx(picp_value, tol) == 0.5


def test_picp_no_coverage() -> None:
    bounds = pd.DataFrame(
        {"lower": [0.0, 0.0, 0.0, 0.0], "upper": [1.0, 1.0, 1.0, 1.0]}, index=index
    )
    picp_value = ps.stats.metrics.picp(obs=obs, bounds=bounds)
    assert pytest.approx(picp_value, tol) == 0.0


def test_picp_mismatched_index() -> None:
    bounds = pd.DataFrame(
        {"lower": [9.0, 14.0, 19.0, 24.0], "upper": [11.0, 16.0, 21.0, 26.0]},
        index=pd.date_range("2010-01-02", "2010-01-05", freq="D"),
    )
    with pytest.raises(ValueError):
        ps.stats.metrics.picp(obs=obs, bounds=bounds)
