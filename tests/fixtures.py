import pytest
from pandas import Series, read_csv, to_datetime

import pastas as ps

rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze("columns")
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze("columns")
obs = (
    read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .dropna()
)


@pytest.fixture
def ml_empty() -> ps.Model:
    ml_empty = ps.Model(obs, name="Test_Model")
    return ml_empty


@pytest.fixture
def ml() -> ps.Model:
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    ml.add_stressmodel(sm)
    return ml


@pytest.fixture
def ml_sm() -> ps.Model:
    ml_sm = ps.Model(obs.dropna(), name="Test_Model")
    sm1 = ps.StressModel(rain, rfunc=ps.Exponential(), name="prec", settings="prec")
    sm2 = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap", settings="evap")
    ml_sm.add_stressmodel([sm1, sm2])
    return ml_sm


@pytest.fixture
def ml_no_settings() -> ps.Model:
    ml_no_settings = ps.Model(obs.dropna(), name="Test_Model")
    sm1 = ps.StressModel(rain, rfunc=ps.Exponential(), name="prec")
    sm2 = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap")
    ml_no_settings.add_stressmodel([sm1, sm2])
    return ml_no_settings


@pytest.fixture
def rm() -> ps.RechargeModel:
    rm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Gamma(), name="rch")
    return rm


@pytest.fixture
def sm_prec() -> ps.StressModel:
    sm_prec = ps.StressModel(rain, rfunc=ps.Exponential(), name="prec")
    return sm_prec


@pytest.fixture
def sm_evap() -> ps.StressModel:
    sm_evap = ps.StressModel(evap, rfunc=ps.Exponential(), name="evap")
    return sm_evap


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
