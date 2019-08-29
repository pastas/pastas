from pandas import read_csv

import pastas as ps


def create_model():
    obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
                   squeeze=True)
    rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                          name='recharge')
    ml.add_stressmodel(sm)
    return ml


def test_least_squares():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares)
    return True


def test_fit_constant():
    ml = create_model()
    ml.solve(fit_constant=False)


def test_no_noise():
    ml = create_model()
    ml.solve(noise=False)
