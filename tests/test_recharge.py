from pandas import read_csv

import pastas as ps


def test_create_rechargemodel():
    rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    rm = ps.RechargeModel(prec=rain, evap=evap, name='recharge')
    return rm


def test_create_model():
    obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
                   squeeze=True)
    ml = ps.Model(obs, name="Test_Model")
    sm = test_create_rechargemodel()
    ml.add_stressmodel(sm)
    return ml


def test_model_solve():
    ml = test_create_model()
    ml.solve()
    return


def test_model_copy():
    ml = test_create_model()
    ml.copy()
    return
