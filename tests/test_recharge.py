from numpy import isclose
from pandas import Series

import pastas as ps


def test_berendrecht(ml_empty: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Berendrecht())
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate()


def test_peterson(ml: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Peterson())
    ml.add_stressmodel(rm)
    ml.simulate()


def test_linear(ml_empty: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Linear())
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate()


def test_flexmodel(ml_empty: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.FlexModel())
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate()


def test_flexmodel_no_interception(
    ml_empty: ps.Model, prec: Series, evap: Series
) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, recharge=ps.rch.FlexModel(interception=False)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate()

def test_flexmodel_gw_uptake(ml_empty: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, recharge=ps.rch.FlexModel(gw_uptake=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate()


def test_flexmodel_snow(
    ml_empty: ps.Model, prec: Series, evap: Series, temp: Series
) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, temp=temp, recharge=ps.rch.FlexModel(snow=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.simulate(tmin=evap.index[0], tmax=evap.index[100])


def test_flexmodel_water_balance_rootzone(prec: Series, evap: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    e = evap.to_numpy()
    sr, r, ea, q, pe = rch.get_root_zone_balance.py_func(p, e)
    error = sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_snow(prec: Series, temp: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    t = temp.to_numpy()
    ss, snow, m = rch.get_snow_balance.py_func(p, t)
    error = ss[0] - ss[-1] + (snow + m)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_interception(prec: Series, evap: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    e = evap.to_numpy()
    si, ei, pi = rch.get_interception_balance.py_func(p, e)
    error = si[0] - si[-1] + (pi + ei)[:-1].sum()
    assert isclose(error, 0)