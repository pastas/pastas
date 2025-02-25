from numpy import arange, isclose, sin
from pandas import Series

import pastas as ps


def test_model_solve(ml: ps.Model) -> None:
    ml.solve()


def test_model_copy(ml: ps.Model) -> None:
    ml.copy()


def test_berendrecht(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, pevap=pevap, recharge=ps.rch.Berendrecht())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_peterson(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, pevap=pevap, recharge=ps.rch.Peterson())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_linear(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, pevap=pevap, recharge=ps.rch.Linear())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, pevap=pevap, recharge=ps.rch.FlexModel())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_no_interception(
    ml_empty: ps.Model, prec: Series, pevap: Series
) -> None:
    rm = ps.RechargeModel(
        prec=prec, pevap=pevap, recharge=ps.rch.FlexModel(interception=False)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_gw_uptake(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    rm = ps.RechargeModel(
        prec=prec, pevap=pevap, recharge=ps.rch.FlexModel(gw_uptake=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_snow(ml_empty: ps.Model, prec: Series, pevap: Series) -> None:
    temp = Series(
        index=pevap.index, data=sin(arange(pevap.size) / 365 * 6), dtype=float
    )
    rm = ps.RechargeModel(
        prec=prec, pevap=pevap, temp=temp, recharge=ps.rch.FlexModel(snow=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_water_balance_rootzone(prec: Series, pevap: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    e = pevap.to_numpy()
    sr, r, ea, q, pe = rch.get_root_zone_balance(p, e)
    error = sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_snow(prec: Series, pevap: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    temp = Series(
        index=pevap.index, data=sin(arange(pevap.size) / 365 * 6), dtype=float
    )
    t = temp.to_numpy()
    ss, snow, m = rch.get_snow_balance(p, t)
    error = ss[0] - ss[-1] + (snow + m)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_interception(prec: Series, pevap: Series) -> None:
    rch = ps.rch.FlexModel()
    p = prec.to_numpy()
    e = pevap.to_numpy()
    si, ei, pi = rch.get_interception_balance(p, e)
    error = si[0] - si[-1] + (pi + ei)[:-1].sum()
    assert isclose(error, 0)
