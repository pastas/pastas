import pastas as ps
from pandas import read_csv

# Load series before
rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                squeeze=True)
evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                squeeze=True)
obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
               squeeze=True)


def test_create_rechargemodel():
    rm = ps.RechargeModel(prec=rain, evap=evap)
    return rm


def test_default_model():
    ml = ps.Model(obs, name="rch_model")
    rm = test_create_rechargemodel()
    ml.add_stressmodel(rm)
    return ml


def test_model_solve():
    ml = test_default_model()
    ml.solve()
    return


def test_model_copy():
    ml = test_default_model()
    ml.copy()
    return


def test_flexmodel():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.FlexModel())
    ml.add_stressmodel(rm)
    ml.solve()
    return


def test_berendrecht():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.Berendrecht())
    ml.add_stressmodel(rm)
    ml.solve()
    return


def test_linear():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.Linear())
    ml.add_stressmodel(rm)
    ml.solve()
    return
