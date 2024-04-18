import pandas as pd
import pytest

import pastas as ps

index = pd.date_range("2010-01-01", "2010-01-04", freq="D")
obs = pd.Series([10.0, 15.21, 20.0, 25.2], index=index)
sim = pd.Series([10.1, 15.0, 20.3, 25.0], index=index)
tol = 1e-6


def test_mae():
    mae = ps.stats.metrics.mae(obs=obs, sim=sim)
    assert pytest.approx(mae, tol) == 0.2025


def test_rmse():
    rmse = ps.stats.metrics.rmse(obs=obs, sim=sim)
    assert pytest.approx(rmse, tol) == 0.2145343


def test_sse():
    sse = ps.stats.metrics.sse(obs=obs, sim=sim)
    assert pytest.approx(sse, tol) == 0.1841


def test_pearsonr():
    r = ps.stats.metrics.pearsonr(obs=obs, sim=sim)
    assert pytest.approx(r, tol) == 0.999299


def test_evp():
    evp = ps.stats.metrics.evp(obs=obs, sim=sim)
    assert pytest.approx(evp, tol) == 99.85505


def test_nse():
    nse = ps.stats.metrics.nse(obs=obs, sim=sim)
    assert pytest.approx(nse, tol) == 0.99855


def test_rsq():
    rsq = ps.stats.metrics.rsq(obs=obs, sim=sim)
    assert pytest.approx(rsq, tol) == 0.99855


def test_bic():
    bic = ps.stats.metrics.bic(obs=obs, sim=sim)
    assert pytest.approx(bic, tol) == -10.9279878


def test_aic():
    aic = ps.stats.metrics.aic(obs=obs, sim=sim)
    assert pytest.approx(aic, tol) == -10.314282


def test_aicc():
    aicc = ps.stats.metrics.aicc(obs=obs, sim=sim)
    assert pytest.approx(aicc, tol) == -8.314282


def test_kge():
    kge = ps.stats.metrics.kge(obs=obs, sim=sim)
    assert pytest.approx(kge, tol) == 0.9923303


def test_kge_modified():
    kgem = ps.stats.metrics.kge(obs=obs, sim=sim, modified=True)
    assert pytest.approx(kgem, tol) == 0.99247
