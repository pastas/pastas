from pandas import read_csv

import pastas as ps


def test_model():
    obs = read_csv("tests/data/obs.csv", index_col=0, parse_dates=True,
                   squeeze=True)
    rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True,
                    squeeze=True)
    evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True,
                    squeeze=True)

    # Create the time series model
    ml = ps.Model(obs, name="Test_Model")

    ## Create stress
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential,
                          name='recharge')
    ml.add_stressmodel(sm)

    # Solve the time series model
    ml.solve()

    return ml


def test_least_squares():
    ml = test_model()
    ml.solve(solver=ps.LeastSquares)
    return True
