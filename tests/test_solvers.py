import pastas as ps

from .fixtures import ml


def test_least_squares(ml):
    ml.solve(solver=ps.LeastSquares())


def test_fit_constant(ml):
    ml.solve(fit_constant=False)


def test_no_noise(ml):
    ml.solve(noise=False)


# test the uncertainty method here
def test_pred_interval(ml):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.prediction_interval(n=10)


def test_ci_simulation(ml):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_simulation(n=10)


def test_ci_block_response(ml):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_block_response(name="rch", n=10)


def test_ci_step_response(ml):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_step_response(name="rch", n=10)


def test_ci_contribution(ml):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_contribution(name="rch", n=10)
