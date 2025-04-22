from pathlib import Path

import pytest
from numpy import arange, sin
from pandas import Series, read_csv, to_datetime

import pastas as ps

data_path = Path(__file__).parent / "data"


def get_prec():
    prec = read_csv(data_path / "rain.csv", index_col=0, parse_dates=True).squeeze()
    return prec


def get_evap():
    evap = read_csv(data_path / "evap.csv", index_col=0, parse_dates=True).squeeze()
    return evap


def get_head() -> Series:
    head = (
        read_csv(data_path / "obs.csv", index_col=0, parse_dates=True)
        .squeeze()
        .dropna()
    )
    return head


@pytest.fixture
def prec() -> Series:
    return get_prec()


@pytest.fixture
def evap() -> Series:
    return get_evap()


@pytest.fixture
def temp() -> Series:
    index = (
        read_csv(data_path / "evap.csv", index_col=0, parse_dates=True).squeeze().index
    )
    return Series(index=index, data=sin(arange(index.size) / 2200), dtype=float)


@pytest.fixture
def head() -> Series:
    return get_head()


@pytest.fixture
def rm(prec: Series, evap: Series) -> ps.RechargeModel:
    rm = ps.RechargeModel(prec=prec, evap=evap, rfunc=ps.Gamma(), name="rch")
    return rm


@pytest.fixture
def sm_prec(prec: Series) -> ps.StressModel:
    sm_prec = ps.StressModel(prec, rfunc=ps.Exponential(), name="prec", settings="prec")
    return sm_prec


@pytest.fixture
def sm_evap(evap: Series) -> ps.StressModel:
    sm_evap = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap", settings="evap")
    return sm_evap


@pytest.fixture
def ml_empty(head: Series) -> ps.Model:
    ml_empty = ps.Model(head, name="Test_Model")
    return ml_empty


@pytest.fixture
def ml_rm(ml_empty: ps.Model, rm: ps.RechargeModel) -> ps.Model:
    ml_empty.add_stressmodel(rm)
    return ml_empty


@pytest.fixture
def ml_sm(
    ml_noise_only: ps.Model, sm_prec: ps.StressModel, sm_evap: ps.StressModel
) -> ps.Model:
    ml_noise_only.add_stressmodel([sm_prec, sm_evap])
    return ml_noise_only


@pytest.fixture
def ml_noise_only(ml_empty: ps.NoiseModel) -> ps.Model:
    ml_empty.add_noisemodel(ps.ArNoiseModel())
    return ml_empty


@pytest.fixture
def ml(ml_rm: ps.Model) -> ps.Model:
    ml = ml_rm.copy()
    ml.add_noisemodel(ps.ArNoiseModel())
    return ml


@pytest.fixture
def ml_solved(ml: ps.Model) -> ps.Model:
    ml.solve()
    return ml


@pytest.fixture
def collwell_data() -> Series:
    """Example Tree C from the publication."""
    n = 9
    x = (
        ["200{}-04-01".format(t) for t in range(0, n)]
        + ["200{}-08-02".format(t) for t in range(0, n)]
        + ["201{}-08-01".format(t) for t in range(0, n)]
    )
    y = [1] * n + [2] * n + [3] * n
    return Series(y, index=to_datetime(x))
