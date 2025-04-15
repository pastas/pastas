import numpy as np
import pandas as pd
import pytest

from pastas.recharge import Linear
from pastas.rfunc import Exponential, One
from pastas.stressmodels import (
    Constant,
    LinearTrend,
    RechargeModel,
    StepModel,
    StressModel,
    WellModel,
)


@pytest.fixture
def setup_data():
    index = pd.date_range("2000-01-01", periods=100, freq="D")
    stress = pd.Series(data=np.random.rand(100), index=index)
    rfunc = Exponential()
    name = "test_stress"
    return index, stress, rfunc, name


def test_stressmodel_simulate(setup_data):
    _, stress, rfunc, name = setup_data
    sm = StressModel(stress, rfunc, name=name)
    params = sm.parameters.initial.values
    result = sm.simulate(params)
    assert len(result) == len(stress)
    assert isinstance(result, pd.Series)


def test_stepmodel_simulate(setup_data):
    index, _, _, name = setup_data
    sm = StepModel(tstart="2000-02-01", name=name, rfunc=One())
    params = sm.parameters.initial.values
    result = sm.simulate(params, tmin="2000-01-01", tmax="2000-04-01", freq="D")
    assert len(result) == len(index)
    assert isinstance(result, pd.Series)


def test_lineartrend_simulate(setup_data):
    index, _, _, name = setup_data
    sm = LinearTrend(start="2000-01-01", end="2000-04-01", name=name)
    params = sm.parameters.initial.values
    result = sm.simulate(params, tmin="2000-01-01", tmax="2000-04-01", freq="D")
    assert len(result) == len(index)
    assert isinstance(result, pd.Series)


def test_constant_simulate(setup_data):
    _, _, _, name = setup_data
    sm = Constant(name=name, initial=5.0)
    result = sm.simulate(p=5.0)
    assert result == 5.0


def test_wellmodel_simulate(setup_data):
    index, stress, _, name = setup_data
    distances = [10, 20]
    stresses = [stress, stress * 2]
    sm = WellModel(stress=stresses, name=name, distances=distances)
    params = sm.parameters.initial.values
    result = sm.simulate(params, tmin="2000-01-01", tmax="2000-04-01", freq="D")
    assert len(result) == len(index)
    assert isinstance(result, pd.Series)


def test_rechargemodel_simulate(setup_data):
    index, stress, rfunc, name = setup_data
    evap = pd.Series(data=np.random.rand(100), index=index)
    sm = RechargeModel(prec=stress, evap=evap, rfunc=rfunc, recharge=Linear())
    params = sm.parameters.initial.values
    result = sm.simulate(params, tmin="2000-01-01", tmax="2000-04-01", freq="D")
    assert len(result) == len(index)
    assert isinstance(result, pd.Series)
