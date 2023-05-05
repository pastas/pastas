from numpy import arange, isclose, sin
from pandas import Series, read_csv

import pastas as ps

# Load series before
rain = (
    read_csv("tests/data/rain.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .loc["2005":]
    * 1e3
)
evap = (
    read_csv("tests/data/evap.csv", index_col=0, parse_dates=True)
    .squeeze("columns")
    .loc["2005":]
    * 1e3
)
temp = Series(index=evap.index, data=sin(arange(evap.size) / 365 * 6), dtype=float)


def test_model_solve(ml: ps.Model) -> None:
    ml.solve()


def test_model_copy(ml: ps.Model) -> None:
    ml.copy()


def test_berendrecht(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.Berendrecht())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_peterson(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.Peterson())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_linear(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.Linear())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(prec=rain, evap=evap, recharge=ps.rch.FlexModel())
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_no_interception(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(
        prec=rain, evap=evap, recharge=ps.rch.FlexModel(interception=False)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_gw_uptake(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(
        prec=rain, evap=evap, recharge=ps.rch.FlexModel(gw_uptake=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_snow(ml_empty: ps.Model) -> None:
    rm = ps.RechargeModel(
        prec=rain, evap=evap, temp=temp, recharge=ps.rch.FlexModel(snow=True)
    )
    ml_empty.add_stressmodel(rm)
    ml_empty.solve()


def test_flexmodel_water_balance_rootzone() -> None:
    rch = ps.rch.FlexModel()
    e = evap.to_numpy()
    p = rain.to_numpy()
    sr, r, ea, q, pe = rch.get_root_zone_balance(p, e)
    error = sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_snow() -> None:
    rch = ps.rch.FlexModel()
    p = rain.to_numpy()
    t = temp.to_numpy()
    ss, snow, m = rch.get_snow_balance(p, t)
    error = ss[0] - ss[-1] + (snow + m)[:-1].sum()
    assert isclose(error, 0)


def test_flexmodel_water_balance_interception() -> None:
    rch = ps.rch.FlexModel()
    e = evap.to_numpy()
    p = rain.to_numpy()
    si, ei, pi = rch.get_interception_balance(p, e)
    error = si[0] - si[-1] + (pi + ei)[:-1].sum()
    assert isclose(error, 0)
