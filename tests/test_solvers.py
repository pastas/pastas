from pandas import read_csv

import pastas as ps


def create_model():
    obs = (
        read_csv("tests/data/obs.csv", index_col=0, parse_dates=True)
        .squeeze("columns")
        .dropna()
    )
    rain = read_csv("tests/data/rain.csv", index_col=0, parse_dates=True).squeeze(
        "columns"
    )
    evap = read_csv("tests/data/evap.csv", index_col=0, parse_dates=True).squeeze(
        "columns"
    )
    ml = ps.Model(obs, name="Test_Model")
    sm = ps.RechargeModel(prec=rain, evap=evap, rfunc=ps.Exponential(), name="recharge")
    ml.add_stressmodel(sm)
    return ml


def test_least_squares():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())


def test_fit_constant():
    ml = create_model()
    ml.solve(fit_constant=False)


def test_no_noise():
    ml = create_model()
    ml.solve(noise=False)


# test the uncertainty method here
def test_pred_interval():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())
    ml.fit.prediction_interval(n=10)


def test_ci_simulation():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_simulation(n=10)


def test_ci_block_response():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_block_response(name="recharge", n=10)


def test_ci_step_response():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_step_response(name="recharge", n=10)


def test_ci_contribution():
    ml = create_model()
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_contribution(name="recharge", n=10)
