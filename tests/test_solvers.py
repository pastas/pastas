import pastas as ps


def test_least_squares(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())


def test_fit_constant(ml: ps.Model):
    ml.solve(fit_constant=False)


def test_no_noise(ml: ps.Model):
    ml.solve(noise=False)


# test the uncertainty method here
def test_pred_interval(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.prediction_interval(n=10)


def test_ci_simulation(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_simulation(n=10)


def test_ci_block_response(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_block_response(name="rch", n=10)


def test_ci_step_response(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_step_response(name="rch", n=10)


def test_ci_contribution(ml: ps.Model):
    ml.solve(solver=ps.LeastSquares())
    ml.fit.ci_contribution(name="rch", n=10)
