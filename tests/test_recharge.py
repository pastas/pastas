from pandas import read_csv, Series
from numpy import sin, arange, isclose
import pastas as ps

# Load series before
rain = read_csv("tests/data/rain.csv", index_col=0,
                parse_dates=True).squeeze("columns").loc["2005":] * 1e3
evap = read_csv("tests/data/evap.csv", index_col=0,
                parse_dates=True).squeeze("columns").loc["2005":] * 1e3
obs = read_csv("tests/data/obs.csv", index_col=0,
               parse_dates=True).squeeze("columns")
temp = Series(index=evap.index, data=sin(arange(evap.size) / 365 * 6),
              dtype=float)


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


def test_flexmodel():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.FlexModel())
    ml.add_stressmodel(rm)
    ml.solve()
    return


def test_flexmodel_no_interception():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap,
                          recharge=ps.rch.FlexModel(interception=False))
    ml.add_stressmodel(rm)
    ml.solve()
    return


def test_flexmodel_snow():
    ml = ps.Model(obs, name="rch_model")
    rm = ps.RechargeModel(prec=rain, evap=evap, temp=temp,
                          recharge=ps.rch.FlexModel(snow=True))
    ml.add_stressmodel(rm)
    ml.solve()
    return


def test_flexmodel_water_balance_rootzone():
    rch = ps.rch.FlexModel()
    e = evap.to_numpy()
    p = rain.to_numpy()
    sr, r, ea, q, pe = rch.get_root_zone_balance(p, e)
    error = (sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum())
    assert isclose(error, 0)


def test_flexmodel_water_balance_snow():
    rch = ps.rch.FlexModel()
    p = rain.to_numpy()
    t = temp.to_numpy()
    ss, snow, m = rch.get_snow_balance(p, t)
    error = (ss[0] - ss[-1] + (snow + m)[:-1].sum())
    assert isclose(error, 0)


def test_flexmodel_water_balance_interception():
    rch = ps.rch.FlexModel()
    e = evap.to_numpy()
    p = rain.to_numpy()
    si, ei, pi = rch.get_interception_balance(p, e)
    error = (si[0] - si[-1] + (pi + ei)[:-1].sum())
    assert isclose(error, 0)
