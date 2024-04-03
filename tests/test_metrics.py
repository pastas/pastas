import pandas as pd
import pytest

import pastas as ps

obs = pd.Series([10.0, 15.0, 20.0, 25.0])
sim = pd.Series([12.0, 18.0, 22.0, 28.0])
tol = 1e-6


def test_mae():
    mae = ps.stats.metrics.mae(obs=obs, sim=sim)
    assert pytest.approx(mae, tol) == 2.5


def test_rmse():
    rmse = ps.stats.metrics.rmse(obs=obs, sim=sim)
    assert pytest.approx(rmse, tol) == 2.549509


def test_sse():
    sse = ps.stats.metrics.sse(obs=obs, sim=sim)
    assert pytest.approx(sse, tol) == 26.0


def test_pearsonr():
    r = ps.stats.metrics.pearsonr(obs=obs, sim=sim)
    assert pytest.approx(r, tol) == 0.997054


def test_evp():
    evp = ps.stats.metrics.evp(obs=obs, sim=sim)
    assert pytest.approx(evp, tol) == 99.2


def test_nse():
    nse = ps.stats.metrics.nse(obs=obs, sim=sim)
    assert pytest.approx(nse, tol) == 0.792


def test_rsq():
    rsq = ps.stats.metrics.rsq(obs=obs, sim=sim)
    assert pytest.approx(rsq, tol) == 0.792


def test_bic():
    bic = ps.stats.metrics.bic(obs=obs, sim=sim)
    assert pytest.approx(bic, tol) == 8.873503


def test_aic():
    aic = ps.stats.metrics.aic(obs=obs, sim=sim)
    assert pytest.approx(aic, tol) == 9.487208


def test_kge():
    kge = ps.stats.metrics.kge(obs=obs, sim=sim)
    assert pytest.approx(kge, tol) == 0.850761


def test_kge_modified():
    kgem = ps.stats.metrics.kge(obs=obs, sim=sim, modified=True)
    assert pytest.approx(kgem, tol) == 0.832548
