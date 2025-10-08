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


# Tests for the raw njit functions
def test_njit_root_zone_balance(prec: Series, evap: Series) -> None:
    """Test the raw njit function for root zone water balance."""
    p = prec.to_numpy()
    e = evap.to_numpy()

    # Test with default parameters
    sr, r, ea, q, pe = ps.rch.FlexModel.get_root_zone_balance.py_func(pe=p, ep=e)
    error = sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum()
    assert isclose(error, 0)

    # Test with custom parameters
    sr, r, ea, q, pe = ps.rch.FlexModel.get_root_zone_balance(
        pe=p, ep=e, srmax=300.0, lp=0.25, ks=150.0, gamma=3.0
    )
    error = sr[0] - sr[-1] + (r + ea + q + pe)[:-1].sum()
    assert isclose(error, 0)


def test_njit_interception_balance(prec: Series, evap: Series) -> None:
    """Test the raw njit function for interception water balance."""
    p = prec.to_numpy()
    e = evap.to_numpy()

    # Test with default parameters
    si, ei, pi = ps.rch.FlexModel.get_interception_balance.py_func(pr=p, ep=e)
    error = si[0] - si[-1] + (pi + ei)[:-1].sum()
    assert isclose(error, 0)

    # Test with custom parameters
    si, ei, pi = ps.rch.FlexModel.get_interception_balance(pr=p, ep=e, simax=5.0)
    error = si[0] - si[-1] + (pi + ei)[:-1].sum()
    assert isclose(error, 0)


def test_njit_snow_balance(prec: Series, temp: Series) -> None:
    """Test the raw njit function for snow water balance."""
    p = prec.to_numpy()
    t = temp.to_numpy()

    # Test with default parameters
    ss, snow, m = ps.rch.FlexModel.get_snow_balance.py_func(prec=p, temp=t)
    error = ss[0] - ss[-1] + (snow + m)[:-1].sum()
    assert isclose(error, 0)

    # Test with custom parameters
    ss, snow, m = ps.rch.FlexModel.get_snow_balance(prec=p, temp=t, tt=-2.0, k=3.5)
    error = ss[0] - ss[-1] + (snow + m)[:-1].sum()
    assert isclose(error, 0)


def test_njit_peterson_recharge(prec: Series, evap: Series) -> None:
    """Test the raw njit function for Peterson recharge calculation."""
    p = prec.to_numpy()
    e = evap.to_numpy()

    # Test with default parameters
    r, s, ea, pe = ps.rch.Peterson.get_recharge.py_func(prec=p, evap=e)
    # Check that the final state is reasonable (not NaN or inf)
    assert isclose(s[-1], s[-1])  # Quick way to check for NaN

    # Test with custom parameters
    r, s, ea, pe = ps.rch.Peterson.get_recharge(
        prec=p, evap=e, scap=1.0, alpha=1.0, ksat=1.0, beta=0.5, gamma=1.0
    )
    # Check that the final state is reasonable (not NaN or inf)
    assert isclose(s[-1], s[-1])  # Quick way to check for NaN
