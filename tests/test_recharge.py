from numpy import isclose
from pandas import Series

import pastas as ps


def test_berendrecht(ml_basic: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Berendrecht())
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_peterson(ml_basic: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Peterson())
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_linear(ml_basic: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.Linear())
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_flexmodel(ml_basic: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(prec=prec, evap=evap, recharge=ps.rch.FlexModel())
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_flexmodel_no_interception(
    ml_basic: ps.Model, prec: Series, evap: Series
) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, recharge=ps.rch.FlexModel(interception=False)
    )
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_flexmodel_gw_uptake(ml_basic: ps.Model, prec: Series, evap: Series) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, recharge=ps.rch.FlexModel(gw_uptake=True)
    )
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate()


def test_flexmodel_snow(
    ml_basic: ps.Model, prec: Series, evap: Series, temp: Series
) -> None:
    rm = ps.RechargeModel(
        prec=prec, evap=evap, temp=temp, recharge=ps.rch.FlexModel(snow=True)
    )
    ml_basic.add_stressmodel(rm)
    ml_basic.simulate(tmin=evap.index[0], tmax=evap.index[100])


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


def test_check_snow_balance(prec: Series, temp: Series) -> None:
    """Test the check_snow_balance method of FlexModel."""
    rch = ps.rch.FlexModel(snow=True)
    error = rch.check_snow_balance(prec=prec.to_numpy(), temp=temp.to_numpy())
    assert isclose(error, 0)


def test_check_interception_balance(prec: Series, evap: Series) -> None:
    """Test the check_interception_balance method of FlexModel."""
    rch = ps.rch.FlexModel(interception=True)
    error = rch.check_interception_balance(prec=prec.to_numpy(), evap=evap.to_numpy())
    assert isclose(error, 0, atol=5e-3)


def test_check_root_zone_balance(prec: Series, evap: Series) -> None:
    """Test the check_root_zone_balance method of FlexModel."""
    rch = ps.rch.FlexModel()
    error = rch.check_root_zone_balance(prec=prec.to_numpy(), evap=evap.to_numpy())
    assert isclose(error, 0, atol=5e-3)


def test_check_balance_with_parameters(
    prec: Series, evap: Series, temp: Series
) -> None:
    """Test that water balance checks work with custom parameters."""
    rch = ps.rch.FlexModel(snow=True, interception=True)

    # Test snow balance with custom parameters
    snow_error = rch.check_snow_balance(
        prec=prec.to_numpy(),
        temp=temp.to_numpy(),
        tt=-2.0,  # threshold temperature
        k=3.5,  # degree-day factor
    )
    assert isclose(snow_error, 0)

    # Test interception balance with custom parameters
    int_error = rch.check_interception_balance(
        prec=prec.to_numpy(),
        evap=evap.to_numpy(),
        simax=5.0,  # maximum interception storage
    )
    assert isclose(int_error, 0, atol=5e-3)

    # Test root zone balance with custom parameters
    rz_error = rch.check_root_zone_balance(
        prec=prec.to_numpy(),
        evap=evap.to_numpy(),
        srmax=300.0,  # maximum root zone storage
        lp=0.3,  # threshold parameter
        ks=150.0,  # saturated hydraulic conductivity
        gamma=3.0,  # nonlinearity parameter
    )
    assert isclose(rz_error, 0, atol=5e-3)
